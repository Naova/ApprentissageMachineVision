import math
import xml.etree.ElementTree as ET
import pathlib
import time
import sys

def get_center_and_radius(xMin, xMax, yMin, yMax):
    xCenter = math.floor((xMax + xMin)/2)
    yCenter = math.floor((yMax + yMin) /2)
    radius = math.floor(xMax - xCenter)
    return xCenter, yCenter, radius

def read_ball_box_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    
    xMin = float(root[6][4][0].text)
    xMax = float(root[6][4][2].text)

    yMin = float(root[6][4][1].text)
    yMax = float(root[6][4][3].text)

    imgPath = root[1].text

    return xMin, xMax, yMin, yMax, imgPath

# Inspired from Florent the Gran
def get_xml_paths(folder):
    folder = pathlib.Path(folder)
    xml_paths = list(folder.glob('*.xml'))
    xml_count = len(xml_paths)
    return xml_paths

if __name__ == '__main__':
    start_time = time.time()
    #xml_paths = get_xml_paths(r"C:\Users\Fred\Desktop\labelImg-master\labelImg-master\savedFred")
    xml_paths = get_xml_paths(sys.argv[1])
    print("Loading from {}".format(sys.argv[1]))
    infosBallBox = []

    for path in xml_paths:
        xMin, xMax, yMin, yMax,imgPath = read_ball_box_xml(path)
        x,y,radius = get_center_and_radius(xMin,xMax,yMin,yMax)
        infosBallBox.append("{},{},{},{}\n".format(x,y,radius,imgPath))

    f = open("imagesCenter.csv", "w")
    for info in infosBallBox:
        f.write(info)

    f.close()

    elapsed_time = round((time.time() - start_time) *1000,4)
    print("{} xml read in {} ms".format(len(xml_paths), elapsed_time))
