Shadow backend
==============

.. automodule:: xrt.backends.shadow


Modifying input files
---------------------

There are two types of input files in shadow:

1) Of 'Namelist' or 'GFile' type (both are in terminology of shadow). These are
   parameter files which consist of lines ``field = value``. Examples of this 
   type are: `start.xx` and `end.xx`. Such files describe optical elements and 
   two sources: geometrical and bending magnet.
2) Of a non-named type consisting of lines of values, one value per line. 
   Examples are ``xsh_input_source_tmp.inp`` and ``xsh_nphoton_tmp.inp``. Such 
   files describe two other sources: wiggler and undulator.

If you want to run a series of ray tracing studies for variable physical or 
geometrical parameters (e.g. for a variable meridional radius of a focusing 
mirror), you have to find out which parameter in the shadow's input files 
controls the desired variable. The only way for this is to play with the 
parameter in a GUI (I use `shadowVUI 
<http://www.esrf.eu/UsersAndScience/Experiments/TBS/SciSoft/xop2.3/extensions>`_)
and look for changes in the corresponding `start.xx` text file. Once you have 
discovered the needed parameter, you can change it in your Python script. There 
are two functions for this:

--------------------------------------------------------------------------------

.. autofunction:: xrt.backends.shadow.modify_input

--------------------------------------------------------------------------------

.. autofunction:: xrt.backends.shadow.modify_xsh_input

--------------------------------------------------------------------------------

The 1st parameter in the above functions is a simple file name or a list of file
names. If you run with several threads or processes then you must modify all the 
versions of the shadow input file in the directories ``tmp0``, ``tmp1`` etc. 
The following function helps in doing this:

.. autofunction:: xrt.backends.shadow.files_in_tmp_subdirs

Writing a loop generator
------------------------

A sequence of ray tracing runs is controlled by a generator (a function that 
returns by ``yield``) which modifies the shadow input files, optionally specifies
information text panels, define the output file names etc. in a loop. The same 
generator is used for normalization, if requested, when it is (quickly) run in 
the second pass.

Consider an example::

    import xrt.runner as xrtr
    import xrt.plotter as xrtp
    import xrt.backends.shadow as shadow
    plot1 = xrtp.XYCPlot('star.01') #create a plot
    textPanel = plot1.fig.text(0.88, 0.8, '', transform=plot1.fig.transFigure,
      size=14, color='r', ha='center') #create a text field, see matplotlib help
    threads = 2
    start01 = shadow.files_in_tmp_subdirs('start.01', threads)
    def plot_generator():
        for thick in [0, 60, 400, 1000, 1500]: #thickness in um
            shadow.modify_input(start01, ('THICK(1)', str(thick * 1e-4)))
            filename = 'filt%04imum' %thick #output file name without extension
            plot1.saveName = [filename + '.pdf', filename + '.png']
            textPanel.set_text('filter\nthickness\n%s $\mu$m' %thick)
            yield

    xrtr.run_ray_tracing(plot1, repeats=40, generator=plot_generator, 
      threads=threads, globalNorm=True)
