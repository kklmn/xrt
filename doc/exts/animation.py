# -*- coding: utf-8 -*-
__copyright__ = u'2016 Konstantin Klementiev, MIT License'
__date__ = "29 Aug 2021"

import os
import shutil
from docutils import nodes
from docutils.parsers.rst import directives, states, Directive
from PIL import Image


class animation(nodes.General, nodes.Element):
    pass


def bottom_right(argument):
    return directives.choice(argument, ('bottom', 'right'))


class AnimationDirective(Directive):
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'alt': directives.unchanged,  # starts with "&ensp;" !!!
                   # for loc:
                   # "upper-left-corner", "lower-left-corner",
                   # "upper-right-corner", "lower-right-corner"
                   # or e.g."top: -500px; left: 400px;"
                   'loc': directives.unchanged,
                   'width': directives.length_or_unitless,
                   'height': directives.length_or_unitless,
                   'scale': directives.percentage,
                   'widthzoom': directives.length_or_unitless,
                   'heightzoom': directives.length_or_unitless,
                   'scalezoom': directives.percentage,
                   'name': directives.unchanged,
                   'align': directives.unchanged,
                   'target': directives.unchanged_required,
                   'class': directives.class_option}
    has_content = True
    aclass = "thumbnail"

    def run(self):
        if not lapp.builder.format.startswith('html'):
            return [nodes.raw('', '')]
#        if not isinstance(self.state, states.SubstitutionDef):
#            raise self.error(
#                'Invalid context: the "%s" directive can only be used within '
#                'a substitution definition.' % self.name)
        self.assert_has_content()
        uri = self.content[0]
        imdest = os.path.join(lapp.outdir, '_images')
        if not os.path.exists(imdest):
            os.mkdir(imdest)
        if uri.endswith('.*'):
            uri = uri[:-1] + 'png'

        if uri.endswith(('.png', '.gif')):
            try:
                im = Image.open(uri)
            except FileNotFoundError:
                uri = os.path.join(lapp.srcdir, uri).replace('\\', '/')
                im = Image.open(os.path.join(lapp.srcdir, uri))

            defwidthzoom, defheightzoom = im.size
            scale = self.options.get('scale', None)
            scalezoom = self.options.get('scalezoom', None)
            if scalezoom:
                scalezoom /= 100.
                defwidthzoom, defheightzoom = \
                    defwidthzoom*scalezoom, defheightzoom*scalezoom

            if scale:
                scale /= 100.
                defwidth, defheight = defwidthzoom*scale, defheightzoom*scale
            else:
                defwidth, defheight = defwidthzoom//2, defheightzoom//2
            shutil.copy2(uri, imdest)
        else:
            with open(os.path.join(uri, 's_anim.js'), 'r') as f:
                rl = f.readline()
            i0 = rl.find('"blit":')
            rl = rl[i0+10:]
            iN = rl.find(']')
            rect = rl[:iN].split(',')
            defwidth, defheight = int(rect[2]), int(rect[3])
            with open(os.path.join(uri, 'b_anim.js'), 'r') as f:
                rl = f.readline()
            i0 = rl.find('"blit":')
            rl = rl[i0+10:]
            iN = rl.find(']')
            rect = rl[:iN].split(',')
            defwidthzoom, defheightzoom = int(rect[2]), int(rect[3])
            imdest_ani = os.path.join(imdest, os.path.basename(uri))
            if not os.path.exists(imdest_ani):
                os.mkdir(imdest_ani)
            shutil.copy2(os.path.join(uri, 's_anim.js'), imdest_ani)
            shutil.copy2(os.path.join(uri, 'b_anim.js'), imdest_ani)
            shutil.copy2(os.path.join(uri, 's_packed.png'), imdest_ani)
            shutil.copy2(os.path.join(uri, 'b_packed.png'), imdest_ani)

        width = self.options.get('width', defwidth)
        height = self.options.get('height', defheight)
        widthzoom = self.options.get('widthzoom', defwidthzoom)
        heightzoom = self.options.get('heightzoom', defheightzoom)

        size = ''
        if width:
            size += ' width="{0}"'.format(width)
        if height:
            size += ' height="{0}"'.format(height)
#        print size
        sizezoom = ''
        if widthzoom:
            sizezoom += ' width="{0}"'.format(widthzoom)
        if heightzoom:
            sizezoom += ' height="{0}"'.format(heightzoom)
#        print sizezoom

        loc = self.options.get('loc', None)
        align = self.options.get('align', None)
        alignC = ''
        if align is not None:
            if align == 'left':
                alignC = 'align-left'
                if loc is None:
                    loc = "upper-left-corner"
            elif align == 'center':
                alignC = 'align-center'
                if loc is None:
                    loc = "center"
            elif align == 'right':
                alignC = 'align-right'
                if loc is None:
                    loc = "upper-right-corner"
            else:
                if loc is None:
                    loc = "upper-left-corner"
        else:
            alignC = 'align-left'
            if loc is None:
                loc = "upper-left-corner"

        loctop = u'top: 0px; '
        lochor = u'width: {}px; '.format(int(widthzoom))
        ta = 'text-align: left; '
        if 'corner' in loc:
            if 'lower' in loc:
                loctop = u'top: -{0}px; '.format(int(heightzoom-height//2))
            else:
                loctop = u'top: {0}px; '.format(0)
            if 'right' in loc:
                ta = 'text-align: right; '
                lochor += u'right: {0}px; '.format(0)
            else:
                ta = 'text-align: left; '
                lochor += u'left: {0}px; '.format(0)
        elif loc == "center":
            ta = 'text-align: center ;'
            loctop = u'top: 0px; '
            lochor += u'left: 50%; transform: translate(-50%, 0%); '
        locst = u'style="{0} {1} {2}"'.format(loctop, lochor, ta)

        alt = self.options.get('alt', '')
        if alt:
            if alt.startswith("&ensp;") or alt.startswith("&emsp;"):
                # Unicode &ensp; or &emsp;
                alt = '{0}'.format(alt)
            else:
                alt = ''
#        self.options['uri'] = uri
        env = self.state.document.settings.env
        targetid = "animation{0}".format(env.new_serialno('animation'))
        if uri.endswith(('.png', '.gif')):
            text = '<a class="{5}">'\
                '<img class="{6}" src="{0}" {1} />'\
                '<span {4}>{2}<br><img class="{6}" src="{0}" {3}/>'\
                '<br>{2}</span></a>'\
                .format(
                    uri, size, alt, sizezoom, locst, self.aclass, alignC)
        else:
            text = '<a class={6}>'\
                '<script type="text/javascript" src="{0}/s_anim.js"></script>'\
                '<script type="text/javascript" src="{0}/b_anim.js"></script>'\
                '<canvas id="s_{1}" {2}></canvas>'\
                '<script>set_animation("{0}/s_packed.png",s_timeline,"s_{1}")'\
                '</script><span {5}>{3}<canvas id="b_{1}" {4}></canvas>{3}'\
                '<script>set_animation("{0}/b_packed.png",b_timeline,"b_{1}")'\
                '</script></span></a>'.format(
                    uri, targetid, size, alt, sizezoom, locst, self.aclass)
        return [nodes.raw('', text, format='html')]


class AnimationHoverDirective(AnimationDirective):
    aclass = "thumbnailhover"


class VideoDirective(Directive):
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'autoplay': directives.flag,
                   'controls': directives.flag,
                   'loop': directives.flag,
                   'width': directives.length_or_unitless,
                   'height': directives.length_or_unitless}
    has_content = True
    aclass = "video"

    def run(self):
        if not lapp.builder.format.startswith('html'):
            return [nodes.raw('', '')]
        self.assert_has_content()
        uri = self.content[0]
        imdest = os.path.join(lapp.outdir, '_videos')
        if not os.path.exists(imdest):
            os.mkdir(imdest)
        if uri.endswith('.*'):
            uri = uri[:-1] + 'mp4'
        shutil.copy2(uri, imdest)

        width = self.options.get('width', None)
        height = self.options.get('height', None)
        sizeStr = ''
        if width:
            sizeStr += ' width="{0}"'.format(width)
        if height:
            sizeStr += ' height="{0}"'.format(height)

        flagStr = ' '.join([flag for flag in ('autoplay', 'controls', 'loop')
                            if flag in self.options])

        # Chromium browsers do not allow autoplay in most cases.
        # However, muted autoplay is always allowed.
        text = '<video src="{0}" muted {1} {2}></video>'.format(
            uri, sizeStr, flagStr)
        return [nodes.raw('', text, format='html')]


def setup(app):
    global lapp
    lapp = app
    try:
        app.add_css_file("thumbnail.css")
    except AttributeError:
        app.add_stylesheet("thumbnail.css")
    try:
        app.add_js_file("animation.js")
    except AttributeError:
        app.add_javascript("animation.js")

    app.add_node(animation)
    app.add_directive('animation', AnimationDirective)
    app.add_directive('imagezoom', AnimationDirective)
    app.add_directive('animationhover', AnimationHoverDirective)
    app.add_directive('imagezoomhover', AnimationHoverDirective)
    app.add_directive('video', VideoDirective)
    return {'version': '1.1'}   # identifies the version of our extension
