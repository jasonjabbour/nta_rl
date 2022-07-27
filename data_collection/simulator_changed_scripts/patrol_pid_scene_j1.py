#!/usr/bin/env python

import os
import inspect
import sys

# Add path of nta_rl directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(1, parentdir)

from nta_rl.digital_twin_data_collection import DigitalTwinDataCollection

if __name__ == '__main__':
    control_and_collect_data = DigitalTwinDataCollection()


	
	
