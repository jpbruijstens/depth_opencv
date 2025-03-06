import cv2
import modules.processing as processing
import modules.streaming as streaming
import modules.sliders as sliders

# def main():
#     #sliders to create filter based on color
#     sliders.create_hsv_sliders("Filter")
# 
#     for frame, depth_frame, spatial_data, config, spatialCalcConfigInQueue in streaming.depth_stream():
#         if frame is None:
#             # wait for next frame
#             continue
# 
#         # Create your own Algorithm here
#         # use the functions from processing.py with "processing.function_name(first_argument, second_argument, ...)"
#         # for example:
# 
#         # Filter the frame
#         filtered_frame = processing.filter_frame(frame)
# 
#         # show the frames
#         cv2.imshow("RGB", frame)
#         cv2.imshow("Depth", depth_frame)
#         cv2.imshow("Filter", filtered_frame)
# 
#         # Close with 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break


def main():
    #sliders to create filter
    sliders.create_hsv_sliders("Filter")
    for frame, depth_frame, spatial_data, config, spatialCalcConfigInQueue in streaming.depth_stream():
        if frame is None:
            # wait for next frame
            continue

        processed_frame = processing.filter_frame(frame)
        
        grey_mask = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        eroded_mask = processing.apply_erosion(grey_mask, 1)
        dilated_mask = processing.apply_dilation(eroded_mask, 1)
        objects = processing.filter_contours(dilated_mask)

        center, corners = processing.bounding_box(frame, objects, depth_frame)

        depth_frame = processing.depth_of_object(depth_frame, spatial_data, corners, center, spatialCalcConfigInQueue, config)

        # Toon de frames
        cv2.imshow("RGB", frame)
        cv2.imshow("Depth", depth_frame)
        cv2.imshow("Filter", processed_frame)

        # Sluit af met 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
