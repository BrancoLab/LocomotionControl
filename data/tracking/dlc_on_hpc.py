import deeplabcut as dlc

import sys


dlc.analyze_videos(sys.argv[0], [sys.argv[1]], sys.argv[2])
