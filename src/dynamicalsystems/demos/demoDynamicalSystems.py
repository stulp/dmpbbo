## \file demoDynamicalSystems.py
## \author Freek Stulp
## \brief  Visualizes results of demoDynamicalSystems.cpp
## 
## \ingroup Demos
## \ingroup DynamicalSystems

import matplotlib.pyplot as plt
import numpy
import os, sys, subprocess

# Include scripts for plotting
lib_path = os.path.abspath('../plotting')
sys.path.append(lib_path)

from plotDynamicalSystem import plotDynamicalSystem
from plotDynamicalSystemComparison import plotDynamicalSystemComparison


if __name__=='__main__':
    executable = "../../../bin/demoDynamicalSystems"
    
    if (not os.path.isfile(executable)):
        print ""
        print "ERROR: Executable '"+executable+"' does not exist."
        print "Please call 'make install' in the build directory first."
        print ""
        sys.exit(-1);
    
    # See if input directory was passed
    if (len(sys.argv)<2 or len(sys.argv)>3):
        print '\nUsage: '+sys.argv[0]+' <test1> [test2]\n';
        print 'Available test labels are:'
        print '   rungekutta - Use 4th-order Runge-Kutta numerical integration.'
        print '   euler      - Use simple Euler numerical integration.'
        print '   analytical - Compute analytical solution (rather than numerical integration)'
        print '   tau        - Change the time constant "tau"'
        print '   attractor  - Change the attractor state during the integration'
        print '   perturb    - Perturb the system during the integration' 
        print ''
        print 'If you call with two tests, the results of the two are compared in one plot.\n'
        sys.exit()
        
    demo_labels = [];
    for arg in sys.argv[1:]:
      demo_labels.append(str(arg))
        
    
    # Call the executable with the directory to which results should be written
    directory = "/tmp/demoDynamicalSystems"
    command = executable+" "+directory
    for demo_label in demo_labels:
      command += " "+demo_label
     
    print "____________________________________________________________________"
    print command
    subprocess.call(command, shell=True)
    
    figure_number = 1;
    directories = os.listdir(directory) 
    for subdirectory in  directories:
        fig = plt.figure(figure_number)
        figure_number = figure_number+1

        data = numpy.loadtxt(directory+"/"+subdirectory+"/results_"+demo_labels[0]+".txt")
        if (len(demo_labels)==1):
          plotDynamicalSystem(data,[fig.add_subplot(1,2,1), fig.add_subplot(1,2,2)])
          fig.canvas.set_window_title(subdirectory+"  ("+demo_labels[0]+")") 
        else:
          data_compare = numpy.loadtxt(directory+"/"+subdirectory+"/results_"+demo_labels[1]+".txt")
          axs      =  [fig.add_subplot(2,2,1), fig.add_subplot(2,2,2)]
          axs_diff =  [fig.add_subplot(2,2,3), fig.add_subplot(2,2,4)]
          # Bit of a hack... We happen to know that SpringDamperSystem is only second order system
          if (subdirectory == "SpringDamperSystem"):
              axs      =  [fig.add_subplot(2,3,1), fig.add_subplot(2,3,2), fig.add_subplot(2,3,3)]
              axs_diff =  [fig.add_subplot(2,3,4), fig.add_subplot(2,3,5), fig.add_subplot(2,3,6)]
          plotDynamicalSystemComparison(data,data_compare,demo_labels[0],demo_labels[1],axs,axs_diff)
          fig.canvas.set_window_title(subdirectory+"  ("+demo_labels[0]+" vs "+demo_labels[1]+")") 
          axs[1].legend()
          
          
        
    plt.show()

