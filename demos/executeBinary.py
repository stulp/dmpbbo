import subprocess
import os

def executeBinary(executable_name,arguments,print_command=False):
    
    if (not os.path.isfile(executable_name)):
        print("")
        print("ERROR: Executable '"+executable+"' does not exist.")
        print("Please call 'make install' in the build directory first.")
        print("")
        sys.exit(-1);
        
    command = executable_name+" "+arguments
    if print_command:
        print(command)
    
    subprocess.call(command, shell=True)
    
    return command
