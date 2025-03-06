# streaming.py
import depthai as dai
import cv2
import numpy as np


def depth_stream():
    """
    Start stream with depthai OAK-D camera.
    """
    # Create pipeline
    pipeline = dai.Pipeline()
    
    # Define sources and outputs
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)
    color = pipeline.create(dai.node.ColorCamera)
    xoutVideo = pipeline.create(dai.node.XLinkOut)
    
    xoutDepth = pipeline.create(dai.node.XLinkOut)
    xoutSpatialData = pipeline.create(dai.node.XLinkOut)
    xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)
    
    xoutDepth.setStreamName("depth")
    xoutSpatialData.setStreamName("spatialData")
    xinSpatialCalcConfig.setStreamName("spatialCalcConfig")
    xoutVideo.setStreamName("video")
    
    # Properties
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setCamera("left")
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setCamera("right")
    
    color.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    color.setInterleaved(False)
    color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    color.setVideoSize(1920, 1080)
    
    xoutVideo.input.setBlocking(False)
    xoutVideo.input.setQueueSize(1)
    
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setExtendedDisparity(False)
    
    # Config
    topLeft = dai.Point2f(0.4, 0.4)
    bottomRight = dai.Point2f(0.6, 0.6)
    
    config = dai.SpatialLocationCalculatorConfigData()
    config.depthThresholds.lowerThreshold = 100
    config.depthThresholds.upperThreshold = 10000
    calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
    config.roi = dai.Rect(topLeft, bottomRight)
    
    spatialLocationCalculator.inputConfig.setWaitForMessage(False)
    spatialLocationCalculator.initialConfig.addROI(config)
    
    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    color.video.link(xoutVideo.input)
    
    spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
    stereo.depth.link(spatialLocationCalculator.inputDepth)
    
    spatialLocationCalculator.out.link(xoutSpatialData.input)
    xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

    # Start pipeline
    with dai.Device(pipeline) as device:

        # Queues
        video = device.getOutputQueue(name="video", maxSize=4, blocking=False)
        depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
        spatialCalcConfigIn = device.getInputQueue("spatialCalcConfig")
    
        # main loop
        while True:
            inDepth = depth.get()
            depthFrame = inDepth.getFrame()
    
            depth_downscaled = depthFrame[::4]
            if np.all(depth_downscaled == 0):
                min_depth = 0
            else:
                min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
                max_depth = np.percentile(depth_downscaled, 99)
                depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
    
                spatialData = spatialCalcQueue.get().getSpatialLocations()
    
                videoIn = video.get()
                ColorFrame = videoIn.getCvFrame()
                resized = cv2.resize(ColorFrame, (640, 400))

                yield (resized, depthFrameColor, spatialData, config, spatialCalcConfigIn)
