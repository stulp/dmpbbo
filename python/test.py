import sys
sys.path.append('../build_dir/python')

import dmpbbo

dmp = dmpbbo.DmpBbo()
dmp.run(0.5, 51, 25, 1, 0.5, '.')

