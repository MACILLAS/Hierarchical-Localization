import argparse
import math
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pymap3d as pm
import scipy.spatial
import torch
from PIL import Image
from transformations import transformations

#from . import logger
from dateutil import parser
from os import listdir

def get_gps_pos(image_path: Path) -> Tuple[float, float, float, np.ndarray, datetime]:
    img = Image.open(image_path)


    for segment, content in img.applist:
        marker, body = content.split(b'\x00', 1)
        if segment == 'APP1' and marker == b'http://ns.adobe.com/xap/1.0/':

            #print(body.decode('utf-8', errors='ignore'))  # Debugging line to check the content of the body

            root = ET.fromstring(body)

            anafi_lat = root[0][0].find('{http://ns.adobe.com/exif/1.0/}GPSLatitude')
            skydio_lat = root[0][0].find('{https://www.skydio.com/drone-skydio/1.0/}Latitude')

            if anafi_lat is not None:
                lat = anafi_lat.text
                split = lat.split(',')
                lat = float(split[0]) + float(split[1][:-1]) / 60
                if split[1][-1] == 'S':
                    lat *= -1

                long = root[0][0].find('{http://ns.adobe.com/exif/1.0/}GPSLongitude').text
                split = long.split(',')
                long = float(split[0]) + float(split[1][:-1]) / 60
                if split[1][-1] == 'W':
                    long *= -1

                alt = root[0][0].find('{http://ns.adobe.com/exif/1.0/}GPSAltitude').text
                if '/' in alt:
                    split = alt.split('/')
                    alt = float(split[0]) / float(split[1])
                else:
                    alt = float(alt)

                roll = float(root[0][0].find('{http://www.parrot.com/drone-parrot/1.0/}CameraRollDegree').text)
                pitch = float(root[0][0].find('{http://www.parrot.com/drone-parrot/1.0/}CameraPitchDegree').text)
                yaw = float(root[0][0].find('{http://www.parrot.com/drone-parrot/1.0/}CameraYawDegree').text)

                orientation = transformations.euler_matrix(roll * math.pi / 180, pitch * math.pi / 180,
                                                           yaw * math.pi / 180)[:3, :3]

                date = parser.parse(root[0][0].find('{http://ns.adobe.com/exif/1.0/}DateTimeOriginal').text)
            elif skydio_lat is not None:
                lat = float(root[0][0].find('{https://www.skydio.com/drone-skydio/1.0/}Latitude').text)
                long = float(root[0][0].find('{https://www.skydio.com/drone-skydio/1.0/}Longitude').text)
                alt = float(root[0][0].find('{https://www.skydio.com/drone-skydio/1.0/}AbsoluteAltitude').text)

                cam_orient_NED = root[0][0].find('{https://www.skydio.com/drone-skydio/1.0/}CameraOrientationNED')
                roll = float(cam_orient_NED.find('{https://www.skydio.com/drone-skydio/1.0/}Roll').text)
                pitch = float(cam_orient_NED.find('{https://www.skydio.com/drone-skydio/1.0/}Pitch').text)
                yaw = float(cam_orient_NED.find('{https://www.skydio.com/drone-skydio/1.0/}Yaw').text)

                orientation = transformations.euler_matrix(roll * math.pi / 180, pitch * math.pi / 180,
                                                           yaw * math.pi / 180)[:3, :3]

                date = parser.parse(root[0][0].find('{http://ns.adobe.com/xap/1.0/}CreateDate').text)
            else:
                lat = float(root[0][0].get('{http://www.dji.com/drone-dji/1.0/}GpsLatitude'))
                long = float(root[0][0].get('{http://www.dji.com/drone-dji/1.0/}GpsLongitude'))
                alt = float(root[0][0].get('{http://www.dji.com/drone-dji/1.0/}AbsoluteAltitude'))

                flight_roll = float(root[0][0].get('{http://www.dji.com/drone-dji/1.0/}FlightRollDegree'))
                flight_pitch = float(root[0][0].get('{http://www.dji.com/drone-dji/1.0/}FlightPitchDegree'))
                flight_yaw = float(root[0][0].get('{http://www.dji.com/drone-dji/1.0/}FlightYawDegree'))

                gimbal_roll = float(root[0][0].get('{http://www.dji.com/drone-dji/1.0/}GimbalRollDegree'))
                gimbal_pitch = float(root[0][0].get('{http://www.dji.com/drone-dji/1.0/}GimbalPitchDegree'))
                gimbal_yaw = float(root[0][0].get('{http://www.dji.com/drone-dji/1.0/}GimbalYawDegree'))

                orientation = (transformations.euler_matrix(flight_roll * math.pi / 180,
                                                            flight_pitch * math.pi / 180,
                                                            flight_yaw * math.pi / 180)
                               @ transformations.euler_matrix(gimbal_roll * math.pi / 180,
                                                              gimbal_pitch * math.pi / 180,
                                                              gimbal_yaw * math.pi / 180))[:3, :3]

                date = parser.parse(root[0][0].get('{http://ns.adobe.com/xap/1.0/}CreateDate'))
                #date = parser.parse(img._getexif()[36867])
                #print(date)

            found = True
            break
    if not found:
        raise Exception('Did not find metadata for {}'.format(image_path))

    return lat, long, alt, orientation, date


def main(output,
         image_dir: Path,
         image_list: List[str],
         closest_geo: int,
         closest_time: int):
    Rs = []
    ts = []
    dates = []
    ref_lat = None
    ref_long = None
    ref_alt = None
    for image_id in image_list:
        lat, long, alt, R, date = get_gps_pos(image_dir / image_id)
        if ref_lat is None:
            ref_lat = lat
            ref_long = long
            ref_alt = alt

        Rs.append(torch.FloatTensor(R).unsqueeze(0))
        ts.append(torch.FloatTensor(pm.geodetic2ned(lat, long, alt, ref_lat, ref_long, ref_alt)).unsqueeze(0))
        dates.append(date.timestamp())

    #logger.info(f'Obtaining pairwise distances between {len(image_list)} images...')

    Rs = torch.cat(Rs)
    ts = torch.cat(ts)
    dates = torch.FloatTensor(dates)
    dates = (dates - torch.min(dates)) # shift time to first image

    pos_dist = torch.cdist(ts, ts)
    date_dist = torch.cdist(dates.unsqueeze(-1), dates.unsqueeze(-1), p=1) # use L1 dist
    date_dist = torch.abs(date_dist) # then absolute value to prevent overflow...

    pairs = []
    for i in range(len(image_list)):
        _, closest_pos = torch.topk(pos_dist[i], closest_geo*2, largest=False)
        _, closest_date = torch.topk(date_dist[i], closest_time, largest=False)

        # Rotation matrix has shape: [N, 3, 3]
        R_relative = torch.matmul(Rs[closest_pos].transpose(-2, -1).unsqueeze(1), Rs[i].unsqueeze(0))
        # Calculate angle from trace, shape: [N, N]
        angle = torch.acos(torch.clamp((R_relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2, -1, 1))
        _, idx = torch.sort(angle[:, 0])
        closest_pos = closest_pos[idx[:closest_geo]]

        for j in torch.cat((closest_pos, closest_date)).unique():
            if i == j:
                continue
            pairs.append((image_list[i], image_list[j]))

    #logger.info(f'Found {len(pairs)} pairs.')

    with open(output, 'w') as f:
        f.write('\n'.join(' '.join(p) for p in pairs))


if __name__ == "__main__":
    output = Path("/home/cviss/PycharmProjects/Hierarchical-Localization/outputs/uw_health_0318_SfM/sfm/pairs-gps.txt")
    image_dir = Path("/home/cviss/Desktop/uwhealth_tripole_0318/images")
    #image_dir = Path("/home/cviss/PycharmProjects/GS_Stream/output/Ford_Tower_06_07/images")

    main(output=output, image_dir=image_dir, image_list=listdir(image_dir), closest_geo=10, closest_time=0)