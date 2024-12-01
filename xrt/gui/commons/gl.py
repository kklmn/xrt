# -*- coding: utf-8 -*-
__author__ = "Roman Chernikov, Konstantin Klementiev"
__date__ = "18 Oct 2017"

try:
    from OpenGL import GL, __version__, __name__  # analysis:ignore
    isOpenGL = True
except ImportError:
    isOpenGL = False

if isOpenGL:

    from OpenGL.GL import glRotatef, glMaterialfv, glClearColor, glMatrixMode,\
        glLoadIdentity, glOrtho, glClear, glEnable, glBlendFunc, glIsEnabled,\
        glEnableClientState, glPolygonMode, glGetDoublev, glDisable,\
        glDisableClientState, glRasterPos3f, glPushMatrix, glTranslatef,\
        glScalef, glPopMatrix, glFlush, glVertexPointerf, glColorPointerf,\
        glLineWidth, glDrawArrays, glMap1f, glMapGrid1f, glEvalMesh1,\
        glMap2f, glMapGrid2f, glEvalMesh2, glFinish, glGetFloatv,\
        glGenBuffers, glBindBuffer, glGetAttribLocation,\
        glEnableVertexAttribArray, glVertexAttribPointer, glBufferData,\
        glLightModeli, glLightfv, glGetIntegerv, glColor4f, glVertex3f,\
        glBegin, glEnd, glViewport, glMaterialf, glHint, glPointSize,\
        glAlphaFunc, glDrawElements, glStencilOp, glStencilFunc, glDepthMask,\
        glReadPixelsui, glReadPixels, glBindFramebuffer, glDepthFunc,\
        GL_FRONT_AND_BACK, GL_AMBIENT, GL_DIFFUSE, GL_SPECULAR, GL_EMISSION,\
        GL_FRONT, GL_SHININESS, GL_PROJECTION, GL_MODELVIEW, GL_INT,\
        GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_MULTISAMPLE, GL_BLEND,\
        GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_POINT_SMOOTH, GL_COLOR_ARRAY,\
        GL_LINE, GL_LINE_SMOOTH, GL_LINE_SMOOTH_HINT, GL_NICEST,\
        GL_POLYGON_SMOOTH_HINT, GL_POINT_SMOOTH_HINT, GL_DEPTH_TEST, GL_FILL,\
        GL_NORMAL_ARRAY, GL_NORMALIZE, GL_VERTEX_ARRAY, GL_QUADS,\
        GL_MAP1_VERTEX_3, GL_MAP2_VERTEX_3, GL_MAP2_NORMAL, GL_LIGHTING, GL_POINTS,\
        GL_LIGHT_MODEL_TWO_SIDE, GL_LIGHT0, GL_POSITION, GL_SPOT_DIRECTION,\
        GL_SPOT_CUTOFF, GL_SPOT_EXPONENT, GL_TRIANGLE_FAN, GL_VIEWPORT,\
        GL_LINES, GL_MODELVIEW_MATRIX, GL_PROJECTION_MATRIX, GL_LINE_WIDTH,\
        GL_LINE_STRIP, GL_FLOAT, GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW,\
        GL_PROGRAM_POINT_SIZE, GL_TRIANGLES, GL_TEXTURE_3D, GL_ALPHA_TEST,\
        GL_GREATER, GL_UNSIGNED_SHORT, GL_CONSTANT_ALPHA, GL_SRC_COLOR,\
        GL_STENCIL_BUFFER_BIT, GL_STENCIL_TEST, GL_KEEP, GL_REPLACE, GL_ALWAYS,\
        GL_UNSIGNED_INT, GL_STENCIL_INDEX, GL_RGBA, GL_DEPTH_COMPONENT,\
        GL_READ_FRAMEBUFFER, GL_DRAW_FRAMEBUFFER, GL_NEAREST, GL_POLYGON_SMOOTH,\
        GL_FALSE, GL_LESS, GL_TRUE

    from OpenGL.GLU import gluPerspective, gluLookAt, gluProject

#    from OpenGL.GLUT import glutBitmapCharacter, glutStrokeCharacter,\
#        glutInit, glutInitDisplayMode, glutStrokeWidth,\
#        GLUT_BITMAP_HELVETICA_10,\
#        GLUT_BITMAP_HELVETICA_12, GLUT_BITMAP_HELVETICA_18,\
#        GLUT_BITMAP_TIMES_ROMAN_10, GLUT_BITMAP_TIMES_ROMAN_24,\
#        GLUT_STROKE_ROMAN, GLUT_RGBA, GLUT_DOUBLE, GLUT_DEPTH,\
#        GLUT_STROKE_MONO_ROMAN, GLUT_STENCIL
#
#    from OpenGL.arrays import vbo

