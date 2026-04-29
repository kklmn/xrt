# -*- coding: utf-8 -*-
__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "18 Oct 2017"

try:
    from OpenGL import __version__, __name__  # analysis:ignore
    isOpenGL = True
except ImportError:
    isOpenGL = False

if isOpenGL:

    from OpenGL.GL import (
        glBlendFunc, glClear, glClearColor, glDepthMask, glDisable,
        glDrawArrays, glDrawArraysInstanced, glDrawElements, glEnable,
        glEnableVertexAttribArray, glGetError, glGetIntegerv, glHint,
        glLineWidth, glPolygonMode, glReadPixels, glStencilFunc, glStencilOp,
        glVertexAttribDivisor, glVertexAttribPointer, glViewport,
        GL_ALWAYS, GL_BLEND, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST, GL_FALSE, GL_FILL, GL_FLOAT, GL_FRONT_AND_BACK,
        GL_KEEP, GL_LINE, GL_LINES, GL_LINE_SMOOTH, GL_LINE_SMOOTH_HINT,
        GL_MULTISAMPLE, GL_NICEST, GL_ONE_MINUS_SRC_ALPHA, GL_POINTS,
        GL_PROGRAM_POINT_SIZE, GL_REPLACE, GL_SRC_ALPHA, GL_STENCIL_BUFFER_BIT,
        GL_STENCIL_INDEX, GL_STENCIL_TEST, GL_TRIANGLE_FAN, GL_TRIANGLES,
        GL_TRUE, GL_UNSIGNED_INT, GL_VIEWPORT)

#    from OpenGL import GL

#    from OpenGL.GL import glRotatef, glMaterialfv, glMatrixMode,\
#        glLoadIdentity, glOrtho, glIsEnabled, glEnableClientState,\
#        glGetDoublev, glDisableClientState, glRasterPos3f, glPushMatrix,\
#        glTranslatef, glScalef, glPopMatrix, glFlush, glVertexPointerf,\
#        glColorPointerf, glMap1f, glMapGrid1f, glEvalMesh1, glMap2f,\
#        glMapGrid2f, glEvalMesh2, glFinish, glGetFloatv, glGenBuffers,\
#        glBindBuffer, glGetAttribLocation, glBufferData, glLightModeli,\
#        glLightfv, glColor4f, glVertex3f, glBegin, glEnd, glMaterialf,\
#        glPointSize, glAlphaFunc, glReadPixelsui, glBindFramebuffer,\
#        glDepthFunc, glBindBufferBase, glBindImageTexture, glDispatchCompute,\
#        glMemoryBarrier, glGetBufferSubData, glDeleteBuffers, GL_AMBIENT,\
#        GL_DIFFUSE, GL_SPECULAR, GL_EMISSION, GL_FRONT, GL_SHININESS,\
#        GL_PROJECTION, GL_MODELVIEW, GL_INT, GL_POINT_SMOOTH, GL_COLOR_ARRAY,\
#        GL_POLYGON_SMOOTH_HINT, GL_POINT_SMOOTH_HINT, GL_NORMAL_ARRAY,\
#        GL_NORMALIZE, GL_VERTEX_ARRAY, GL_QUADS, GL_MAP1_VERTEX_3,\
#        GL_MAP2_VERTEX_3, GL_MAP2_NORMAL, GL_LIGHTING,\
#        GL_LIGHT_MODEL_TWO_SIDE, GL_LIGHT0, GL_POSITION, GL_SPOT_DIRECTION,\
#        GL_SPOT_CUTOFF, GL_SPOT_EXPONENT, GL_MODELVIEW_MATRIX,\
#        GL_PROJECTION_MATRIX, GL_LINE_WIDTH, GL_LINE_STRIP, GL_ARRAY_BUFFER,\
#        GL_DYNAMIC_DRAW, GL_TEXTURE_3D, GL_ALPHA_TEST, GL_GREATER,\
#        GL_UNSIGNED_SHORT, GL_CONSTANT_ALPHA, GL_SRC_COLOR, GL_RGBA,\
#        GL_DEPTH_COMPONENT, GL_READ_FRAMEBUFFER, GL_DRAW_FRAMEBUFFER,\
#        GL_NEAREST, GL_POLYGON_SMOOTH, GL_LESS, GL_SHADER_STORAGE_BUFFER,\
#        GL_WRITE_ONLY, GL_RGB32F, GL_SHADER_STORAGE_BARRIER_BIT,\
#        GL_TEXTURE_2D
