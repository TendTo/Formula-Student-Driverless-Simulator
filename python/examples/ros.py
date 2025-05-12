import time
import fsds
from fsds.ros import RosBridgeClient

# connect to the AirSim simulator 
client = fsds.FSDSClient()

# Check network connection
client.confirmConnection()

# After enabling api controll only the api can controll the car. 
# Direct keyboard and joystick into the simulator are disabled.
# If you want to still be able to drive with the keyboard while also 
# controll the car using the api, call client.enableApiControl(False)
client.enableApiControl(False)


with RosBridgeClient() as ros_client:
    ros_client.publish("/fest", msg=fsds.Vector3r(x_val=31, y_val=2, z_val=22))

exit(1)

while True:
    lidardata = client.getLidarData()
    print(lidardata)
    print(lidardata.to_dict())
    time.sleep(2)

