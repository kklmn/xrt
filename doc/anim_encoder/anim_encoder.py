#!/usr/bin/env python
# Copyright (c) 2012, Sublime HQ Pty Ltd
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import scipy.ndimage.measurements as me
import json
import scipy.misc as misc
from PIL import Image
import glob
#import sys
#import os
import cv2
import numpy as np
from time import time

# How long to wait before the animation restarts
END_FRAME_PAUSE = 1000

# How many pixels can be wasted in the name of combining neighbouring changed
# regions.
SIMPLIFICATION_TOLERANCE = 512
MAX_PACKED_HEIGHT = 100000


def PIL2array(img):
    return np.array(img.convert('RGB').getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)


def slice_size(a, b):
    return (a.stop - a.start) * (b.stop - b.start)


def combine_slices(a, b, c, d):
    return (slice(min(a.start, c.start), max(a.stop, c.stop)),
            slice(min(b.start, d.start), max(b.stop, d.stop)))


def slices_intersect(a, b, c, d):
    if (a.start >= c.stop): return False
    if (c.start >= a.stop): return False
    if (b.start >= d.stop): return False
    if (d.start >= b.stop): return False
    return True


# Combine a large set of rectangles into a smaller set of rectangles,
# minimising the number of additional pixels included in the smaller set of
# rectangles
def simplify(boxes, tol=SIMPLIFICATION_TOLERANCE):
    out = []
    for a, b in boxes:
        sz1 = slice_size(a, b)
        did_combine = False
        for i in xrange(len(out)):
            c, d = out[i]
            cu, cv = combine_slices(a, b, c, d)
            sz2 = slice_size(c, d)
            if slices_intersect(a, b, c, d) or \
                    (slice_size(cu, cv) <= sz1 + sz2 + tol):
                out[i] = (cu, cv)
                did_combine = True
                break
        if not did_combine:
            out.append((a, b))

    if tol != 0:
        return simplify(out, 0)
    else:
        return out


def slice_tuple_size(s):
    a, b = s
    return (a.stop - a.start) * (b.stop - b.start)

# Allocates space in the packed image. This does it in a slow, brute force
# manner.
class Allocator2D:
    def __init__(self, rows, cols):
        self.bitmap = np.zeros((rows, cols), dtype=np.uint8)
        self.available_space = np.zeros(rows, dtype=np.uint32)
        self.available_space[:] = cols
        self.num_used_rows = 0

    def allocate(self, w, h):
        bh, bw = self.bitmap.shape

        for row in xrange(bh - h + 1):
            if self.available_space[row] < w:
                continue

            for col in xrange(bw - w + 1):
                if self.bitmap[row, col] == 0:
                    if not self.bitmap[row:row+h, col:col+w].any():
                        self.bitmap[row:row+h, col:col+w] = 1
                        self.available_space[row:row+h] -= w
                        self.num_used_rows = max(self.num_used_rows, row + h)
                        return row, col
        raise RuntimeError()


def find_matching_rect(bitmap, num_used_rows, packed, src, sx, sy, w, h):
    template = src[sy:sy+h, sx:sx+w]
    bh, bw = bitmap.shape
    image = packed[0:num_used_rows, 0:bw]

    if num_used_rows < h:
        return None

    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    row, col = np.unravel_index(result.argmax(), result.shape)
    if ((packed[row:row+h, col:col+w] == src[sy:sy+h, sx:sx+w]).all()
        and (packed[row:row+1, col:col+w, 0] == src[sy:sy+1, sx:sx+w, 0]).all()):
        return row, col
    else:
        return None


def generate_animation(dirname, fname, time_factor=5,
                       isBidirectional=True, slice_frames=slice(None),
                       endFrameDelay=END_FRAME_PAUSE):
    prefix = 's' if fname.startswith('s_') else 'b'
    if not fname.endswith('.png'):
        fname += '*.png'
    print dirname
    print fname
    frames = glob.glob(dirname + r'/' + fname)
    if not frames:
        raise ValueError("no files found!")
    frames = frames[slice_frames]
    lenframes = len(frames)
    print '{0} file{1} found'.format(lenframes, '' if lenframes == 1 else 's')
    for frame in frames:
        print frame
#    images = [misc.imread(f) for f in frames]
    pimages = [Image.open(f) for f in frames]
    images = [PIL2array(img) for img in pimages]
    print images[0].shape
#    if images[0].shape[2] == 4:  # if with alpha channel
#        images3 = []
#        for image in images:
#            r, g, b, a = np.rollaxis(image, axis = -1)
#            r[a == 0] = 255
#            g[a == 0] = 255
#            b[a == 0] = 255
#            images3.append(np.dstack([r, g, b]))
#        images = images3
    ih, iw = images[0].shape[0:2]
    delays = [50*time_factor for f in frames]
    if isinstance(endFrameDelay, (float, int)):
        delays[-1] = endFrameDelay
#    print delays

    not_bkgnd = np.ones_like(images[0]) * 0.5
    pairs = zip([not_bkgnd] + images[:-1], images)
    diffs = [np.sign((bp - ap).max(axis=2)) for ap, bp in pairs]

    # Find different objects for each frame
    img_areas = [me.find_objects(me.label(d)[0]) for d in diffs]
    if isBidirectional:
        for src_rects, copy_rects in zip(img_areas[:-1], img_areas[1:]):
            src_rects += copy_rects
    # Simplify areas
    img_areas = [simplify(x) for x in img_areas]

    # Generate a packed image
    allocator = Allocator2D(MAX_PACKED_HEIGHT, iw)
    packed = np.zeros((MAX_PACKED_HEIGHT, iw, 3), dtype=np.uint8)

    # Sort the rects to be packed by largest size first, to improve the packing
    rects_by_size = []
    for i in xrange(len(images)):
        src_rects = img_areas[i]
        for j in xrange(len(src_rects)):
            rects_by_size.append((slice_tuple_size(src_rects[j]), i, j))

    rects_by_size.sort(reverse=True)

    allocs = [[None] * len(s) for s in img_areas]

    print "packing {0} rects of {1} frames".format(
        len(rects_by_size), len(images))

    t0 = time()

    for ie, (size, i, j) in enumerate(rects_by_size):
        print ie,
        src = images[i]
        src_rects = img_areas[i]

        a, b = src_rects[j]
        sx, sy = b.start, a.start
        w, h = b.stop - b.start, a.stop - a.start

        # See if the image data already exists in the packed image. This takes
        # a long time, but results in worthwhile space savings (20% in one
        # test)
        existing = find_matching_rect(
            allocator.bitmap, allocator.num_used_rows, packed,
            src, sx, sy, w, h)
        if existing:
            dy, dx = existing
            allocs[i][j] = (dy, dx)
        else:
            dy, dx = allocator.allocate(w, h)
            allocs[i][j] = (dy, dx)
            packed[dy:dy+h, dx:dx+w] = src[sy:sy+h, sx:sx+w]

    print
    print "packing finished, took:", time() - t0

    packed = packed[0:allocator.num_used_rows]

#    misc.imsave(dirname + "/_packed_tmp.png", packed)
#    # Don't completely fail if we don't have pngcrush
#    if os.system("pngcrush -q " + dirname + "/_packed_tmp.png " + dirname + "/_packed.png") == 0:
#        os.system("rm " + dirname + "/_packed_tmp.png")
#    else:
#        print "pngcrush not found, output will not be larger"
#        os.system("mv " + dirname + "/_packed_tmp.png " + dirname + "/_packed.png")
    misc.imsave(dirname + r"/{0}_packed.png".format(prefix), packed)

    # Generate JSON to represent the data
    timeline = []
    if isBidirectional:
        img_areas += img_areas[-2::-1]
        allocs += allocs[-2::-1]
        lastdelay = delays[-1]
        delays += delays[-2::-1]
        delays[-1] = lastdelay - delays[0]
    for i, (src_rects, dst_rects, delay) in enumerate(
            zip(img_areas, allocs, delays)):
        blitlist = []
        for j in xrange(len(src_rects)):
            a, b = src_rects[j]
            sx, sy = b.start, a.start
            w, h = b.stop - b.start, a.stop - a.start
            dy, dx = dst_rects[j]
            blitlist.append([int(l) for l in [dx, dy, w, h, sx, sy]])
        timeline.append({'delay': delay, 'blit': blitlist})

#    print timeline
    f = open(dirname + r'/{0}_anim.js'.format(prefix), 'wb')
    f.write(r"{0}_timeline = ".format(prefix))
    json.dump(timeline, f)
    f.close()


if __name__ == '__main__':
#    generate_animation(sys.argv[1], sys.argv[2])

#    generate_animation(r"c:\Ray-tracing\tests\raycing\IpPol", r"s_IpPol",
#                       time_factor=2)
#    generate_animation(r"c:\Ray-tracing\tests\raycing\IpPol", r"b_IpPol",
#                       time_factor=2)

#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\2-VCMPitch",
#                       r"s_vcmSi-FootprintP")
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\2-VCMPitch",
#                       r"vcmSi-FootprintP")

#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\34_Tapering\norm2\out", r"s_",
#                       slice_frames=slice(None, None, 5))
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\34_Tapering\norm2\out", r"b_",
#                       slice_frames=slice(None, None, 5))

#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\1-FilterThickness",
#                       r"s_filterThicknessP")
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\1-FilterThickness",
#                       r"b_filterThicknessP")

#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\1-FilterThickness",
#                       r"s_filterThicknessI")
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\1-FilterThickness",
#                       r"b_filterThicknessI")

#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\2-VCMPitch",
#                       r"s_vcmSi-FSM")
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\2-VCMPitch",
#                       r"vcmSi-FSM")

#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\2-VCMPitch",
#                       r"s_vcmIr-FSM")
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\2-VCMPitch",
#                       r"vcmIr-FSM")

#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\2-VCMPitch",
#                       r"s_vcmIr-FootprintP")
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\2-VCMPitch",
#                       r"vcmIr-FootprintP")

#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\3-VCMR",
#                       r"s_vcmR-DCM")
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\3-VCMR",
#                       r"vcmR-DCM")

#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\3-VCMR",
#                       r"s_vcmR-Sample")
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\3-VCMR",
#                       r"vcmR-Sample")

#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\4-VFMR",
#                       r"s_vfmR-Sample")
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\4-VFMR",
#                       r"vfmR-Sample")

#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\5-mirrorPitch",
#                       r"s_pitch-Sample")
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\5-mirrorPitch",
#                       r"pitch-Sample")

#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\6-DCMScan",
#                       r"s_Si111", time_factor=10)
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\6-DCMScan",
#                       r"Si111", time_factor=10)

#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\6-DCMScan",
#                       r"s_Si311", time_factor=10)
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\02_Balder_BL\6-DCMScan",
#                       r"Si311", time_factor=10)

#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\12_Multilayer",
#                       r"s_1stML_*_norm2.png", time_factor=10)
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\12_Multilayer",
#                       r"1stML_*_norm2.png", time_factor=10)

#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\12_Multilayer",
#                       r"s_afterDMM_*_norm2", time_factor=10)
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\12_Multilayer",
#                       r"afterDMM_*_norm2", time_factor=10)

#    Estr = '36'
#    fname = '01_BentLaueSCM_FSM2_E_hor_*{0}keV_flat.png'.format(Estr)
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\03_LaueMono",
#                       "s_"+fname, time_factor=20, endFrameDelay=2000)
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\03_LaueMono",
#                       fname, time_factor=20, endFrameDelay=2000)

#    fname = '02_bentLaueDCM_FSM2_E_hor_R=025m_09keV*.png'
# needs frames[0:25] = frames[24::-1]
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\03_LaueMono",
#                       "s_"+fname, time_factor=20,
#                       slice_frames=slice(None, None, 5), endFrameDelay=2000)
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\03_LaueMono",
#                       fname, time_factor=20,
#                       slice_frames=slice(None, None, 5), endFrameDelay=2000)

#    Estr = '36'
#    fname = u'rc_R*_{0}keV_flat.png'.format(Estr)
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\03_LaueMono",
#                       "s_"+fname, time_factor=20, endFrameDelay=2000)
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\03_LaueMono",
#                       fname, time_factor=20, endFrameDelay=2000)


#    fname = '01_BentLaueSCM_FSM2_E_hor_*{0}keV_flat.png'.format(Estr)
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\03_LaueMono",
#                       "s_"+fname, time_factor=20, endFrameDelay=2000)
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\03_LaueMono",
#                       fname, time_factor=20, endFrameDelay=2000)

#    fname = 'CRL-block-2-Al-hor-FSM2-*.png'
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\04_Lenses",
#                       "s_"+fname, time_factor=5,
#                       slice_frames=slice(None, None, 5))
#    generate_animation(r"c:\Ray-tracing\examples\withRaycing\04_Lenses",
#                       fname, time_factor=5,
#                       slice_frames=slice(None, None, 5))

#    dirName = r'c:\Ray-tracing\examples\withRaycing\05_QWP'
##'QWP-BT-E'
##    fname = '03_col_BT_FSM2_E_*_norm1.png'
##'QWP-BT-CircPolRate'
##    fname = '03_col_BT_FSM2_CircPolRate_*_norm1.png'
##'QWP-BT-PhaseShift'
##    fname = '03_col_BT_FSM2_PhaseShift_*_norm1.png'
##'QWP-BT-PolAxesRatio'
##    fname = '03_col_BT_FSM2_PolAxesRatio_*_norm1.png'
##'QWP-LT-E'
##    fname = '04_col_LT_FSM2_E_*_norm1.png'
##'QWP-LT-CircPolRate'
##    fname = '04_col_LT_FSM2_CircPolRate_*_norm1.png'
##'QWP-LT-PhaseShift'
##    fname = '04_col_LT_FSM2_PhaseShift_*_norm1.png'
##'QWP-LT-PolAxesRatio'
##    fname = '04_col_LT_FSM2_PolAxesRatio_*_norm1.png'
##'QWP-LT-conv-CircPolRate'
##    fname = '06_conv_LT_FSM2_CircPolRate_*_norm1.png'
##'QWP-LT-conv-bent-CircPolRate'
#    fname = '08_conv_LT_bent_FSM2_CircPolRate_*_norm1.png'
#    generate_animation(dirName, "s_"+fname, time_factor=10)
#    generate_animation(dirName, fname, time_factor=10)

#    dirName = r'c:\Ray-tracing\examples\withRaycing\06_AnalyzerBent1D'
#    fname = '1D-02gb-Si111-*-det_DegOfPol-flat.png'
#    generate_animation(dirName,
#                       "s_"+fname, time_factor=20, endFrameDelay=2000)
#    generate_animation(dirName,
#                       fname, time_factor=20, endFrameDelay=2000)
#    dirName = r'c:\Ray-tracing\examples\withRaycing\06_AnalyzerBent1D'
#    fname = '1D-04lgb-Si111-*-det_DegOfPol-flat.png'
#    generate_animation(dirName, "s_"+fname, time_factor=20, endFrameDelay=2000)
#    generate_animation(dirName, fname, time_factor=20, endFrameDelay=2000)

#    dirName = r'c:\Ray-tracing\examples\withRaycing\09_Gratings'
##    fname = 'FZP-localZ*.png'
##    fname = 'FZP-FSM2_Es*.png'
##    generate_animation(dirName, "s_"+fname, time_factor=10,
##                       isBidirectional=False, endFrameDelay=None)
##    generate_animation(dirName, fname, time_factor=10,
##                       isBidirectional=False, endFrameDelay=None)
#    fname = 'BraggFresnelFSM2-NE*.png'
#    generate_animation(dirName, "s_"+fname, time_factor=10)
#    generate_animation(dirName, fname, time_factor=10)

#    dirName = r'c:\Ray-tracing\examples\withRaycing\09_Gratings\EnergyScan'
##    fname = 'FlexPES-12-FSMExp2Op00*.png'
#    fname = 'FlexPES-08-FSM3vfOp00*.png'
#    generate_animation(dirName, "s_"+fname)
#    generate_animation(dirName, fname)

#    dirName = r'c:\Ray-tracing\examples\withRaycing\09_Gratings\SlitScan'
#    fname = 'FlexPES-12-*.png'
#    generate_animation(dirName, "s_"+fname)
#    generate_animation(dirName, fname)

#    dirName = r'c:\Ray-tracing\examples\withRaycing\41_DoubleSlit'
#    fname = '2 - DS Propagation Rays*.png'
##    fname = '3 - DS Propagation Wave*.png'
#    generate_animation(dirName, "s_"+fname, time_factor=20, endFrameDelay=2000)
#    generate_animation(dirName, fname, time_factor=20, endFrameDelay=2000)

#    dirName = r'c:\Ray-tracing\examples\withRaycing\43_SoftiSTXM\out'
#    fname = 'stxm-2D-1-rays-0emit-0enSpread-monoE-06i-ExpFocus-Is*.png'
#    fname = 'stxm-2D-2-hybr-0emit-0enSpread-monoE-06i-ExpFocus-Is*.png'
#    fname = 'stxm-2D-2-hybr-non0e-0enSpread-monoE-06i-ExpFocus-Is*.png'
#    fname = 'stxm-2D-2-hybr-non0e-0enSpread-wideE-06i-ExpFocus-Is*.png'
#    fname = 'stxm-2D-2-hybr-non0e-0enSpread-monoE-025*06*.png'
#    fname = 'stxm-IDOC-2D-2-hybr-non0e-0enSpread-monoE--??.png'
#    fname = 'stxm-IDOC-2D-2-hybr-non0e-0enSpread-monoE-025*.png'
#    fname = 'stxm-Modes-eigen modes of mutual intensity-2D-2-hybr-non0e-0enSpread-monoE--*.png'
#    fname = 'stxm-Modes-eigen modes of mutual intensity-2D-2-hybr-non0e-0enSpread-monoE-025%H*.png'
#    fname = 'stxm-Modes-principal components of one-electron images-2D-2-hybr-non0e-0enSpread-monoE--*.png'
#    fname = 'stxm-Modes-principal components of one-electron images-2D-2-hybr-non0e-0enSpread-monoE-025%H*.png'

#    dirName = r'c:\Ray-tracing\examples\withRaycing\44_SoftiCXI\1e6'
#    fname = 'cxi-2D-2-hybr-0emit-0enSpread-monoE-08i-ExpFocus-Is*.png'
#    fname = 'cxi-1D-2-hybr-1e6hor-0emit-0enSpread-monoE-08e-ExpFocus*.png'
#    fname = 'cxi-1D-2-hybr-1e6ver-0emit-0enSpread-monoE-08e-ExpFocus*.png'
#    fname = 'cxi-1D-2-hybr-1e6hor-0emit-0enSpread-monoE-08p-ExpFocusPhase*.png'
#    fname = 'cxi-1D-2-hybr-1e6hor-0emit-0enSpread-monoE-08pf-ExpFocusPhaseFront*.png'

#    dirName = r'c:\Ray-tracing\examples\withRaycing\44_SoftiCXI'
##    fname = '1D-1-rays-hor-0emit-0enSpread-monoE-08e-ExpFocus*.png'
#    fname = '1D-2-hybr-hor-0emit-0enSpread-monoE-08e-ExpFocus*.png'
##    fname = '1D-3-wave-hor-0emit-0enSpread-monoE-08e-ExpFocus*.png'
##    fname = 'XXIDOC-1D-2-hybr-hor-0emit-0enSpread-monoE--*.png'
##    fname = 'XXIDOC-1D-2-hybr-hor-non0e-0enSpread-monoE--*.png'
#
#    generate_animation(dirName, "s_"+fname, time_factor=20, endFrameDelay=2000)
#    generate_animation(dirName, fname, time_factor=20, endFrameDelay=2000)

#    dirName = r'c:\Ray-tracing\examples\withRaycing\33_Warping'
#    fname = 'rays-perfect-*.png'
#    fname = 'rays-gaussian-*.png'
#    fname = 'rays-waviness-*.png'
#    fname = 'rays-NOM-*.png'
#    fname = 'wave-non0e-perfect-*.png'
#    fname = 'wave-non0e-gaussian-*.png'
#    fname = 'wave-non0e-waviness-*.png'
#    fname = 'wave-non0e-NOM-*.png'

#    generate_animation(dirName, "s_"+fname, time_factor=20, endFrameDelay=2000)
#    generate_animation(dirName, fname, time_factor=20, endFrameDelay=2000)

#    dirName = r'c:\Ray-tracing\examples\withRaycing\11_Waves'
##    fname = '2 - DS Propagation Rays -*.png'
#    fname = '3 - DS Propagation Wave -*.png'
#
#    generate_animation(dirName, "s_"+fname, time_factor=10, endFrameDelay=1000)
#    generate_animation(dirName, fname, time_factor=10, endFrameDelay=1000)

    dirName = r'c:\Ray-tracing\tests\raycing'
    fname = '_Laguerre-Gauss-*.png'

    generate_animation(dirName, "s_"+fname, endFrameDelay=1000)
    generate_animation(dirName, fname, endFrameDelay=1000)
