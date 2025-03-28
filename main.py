import cv2
import modules.processing as processing
import modules.streaming as streaming
import modules.sliders as sliders


def main():
    #sliders to create filter
    sliders.create_hsv_sliders("Filter")
    frame = None # color frame
    depth_frame = None # depth frame
    spatial_data = None # spatial data, contains the depth data for the complete frame
    config = None # config object for the depth stream
    spatialCalcConfigInQueue = None # spatialCalcConfigInQueue object for the depth stream

    # retrieve frames from the camera with the spatial data and configs
    for frame, depth_frame, spatial_data, config, spatialCalcConfigInQueue in streaming.depth_stream():
        if frame is None:
            # wait for next frame
            continue

        # Use color filter on frame
        processed_frame = processing.filter_frame(frame)
        # convert to grayscale for further processing
        grey_mask = processing.apply_mask(processed_frame)
        # apply erosion and dilation to remove noise
        eroded_mask = processing.apply_erosion(grey_mask)
        dilated_mask = processing.apply_dilation(eroded_mask)
        # find objects in the frame
        objects = processing.filter_contours(dilated_mask)
        # Draw bounding box around objects
        center, corners = processing.bounding_box(frame, objects, depth_frame)
        # Calculate depth of object
        depth_frame = processing.depth_of_object(depth_frame, spatial_data, corners, center, spatialCalcConfigInQueue, config)

        # Show the frames
        cv2.imshow("RGB", frame)
        cv2.imshow("Depth", depth_frame)
        cv2.imshow("Filter", processed_frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
