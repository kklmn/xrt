# -*- coding: utf-8 -*-
__copyright__ = u'2016 Konstantin Klementiev, MIT License'
__date__ = "04 Aug 2017"

import os
import shutil
from docutils import nodes
from docutils.parsers.rst import directives, states
from docutils.parsers.rst import Directive
from PIL import Image


class animation(nodes.General, nodes.Element):
    pass


def bottom_right(argument):
    return directives.choice(argument, ('bottom', 'right'))


class AnimationDirective(Directive):
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'alt': directives.unchanged,
# "upper-left-corner" (def), "lower-left-corner",
# "upper-right-corner", "lower-right-corner" or e.g."top: -500px; left: 400px;"
                   'loc': directives.unchanged,
                   'width': directives.length_or_unitless,
                   'height': directives.length_or_unitless,
                   'scale': directives.percentage,
                   'widthzoom': directives.length_or_unitless,
                   'heightzoom': directives.length_or_unitless,
                   'scalezoom': directives.percentage,
                   'name': directives.unchanged,
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

        if uri.endswith('.png'):
            im = Image.open(uri)
            defwidthzoom, defheightzoom = im.size
            scale = self.options.get('scale', None)
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

        loc = self.options.get('loc', 'upper-left-corner')
        if 'corner' in loc:
            if 'lower' in loc:
                loctop = u'top: -{0}px'.format(int(heightzoom)-20)
            else:
                loctop = u'top: -{0}px'.format(height)
            if 'right' in loc:
                locleft = u'left: -{0}px'.format(int(widthzoom)-int(width))
            else:
                locleft = u'left: -{0}px'.format(0)
            locst = u'style="{0}; {1};"'.format(loctop, locleft)
        else:
            locst = u'style="{0}"'.format(loc)

        alt = self.options.get('alt', '')
        if alt:
            if alt.startswith("&ensp;") or alt.startswith("&ensp;"):
                # Unicode &ensp; or &emsp;
#                alt = '<br />{0}'.format(alt)
                alt = '{0}'.format(alt)
            else: # otherwise the substitution name is passed by docutils as 'alt'
                alt = ''
#        self.options['uri'] = uri
        env = self.state.document.settings.env
        targetid = "animation{0}".format(env.new_serialno('animation'))
        if uri.endswith('.png'):
            text = '<a class={6}>'\
            '<img src="{0}" {1} />'\
            '<span {5}>{2}'\
            '<canvas id="{3}" {4} ></canvas>{2}'\
            '<script>set_static("{0}", "{3}")</script></span></a>'.format(
                uri, size, alt, targetid, sizezoom, locst, self.aclass)
        else:
            text = '<a class={6}>'\
            '<script type="text/javascript" src="{0}/s_anim.js"></script>'\
            '<script type="text/javascript" src="{0}/b_anim.js"></script>'\
            '<canvas id="s_{1}" {2}></canvas>'\
            '<script>set_animation("{0}/s_packed.png", s_timeline, "s_{1}")'\
            '</script><span {5}>{3}<canvas id="b_{1}" {4}></canvas>{3}'\
            '<script>set_animation("{0}/b_packed.png", b_timeline, "b_{1}")'\
            '</script></span></a>'.format(
                uri, targetid, size, alt, sizezoom, locst, self.aclass)
        return [nodes.raw('', text, format='html')]


class AnimationHoverDirective(AnimationDirective):
    aclass = "thumbnailhover"


def setup(app):
    global lapp
    lapp = app
    app.add_node(animation)
    app.add_directive('animation', AnimationDirective)
    app.add_directive('imagezoom', AnimationDirective)
    app.add_directive('animationhover', AnimationHoverDirective)
    app.add_directive('imagezoomhover', AnimationHoverDirective)
    return {'version': '0.1'}   # identifies the version of our extension
