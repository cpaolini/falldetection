#include <librealsense2/rs.hpp>
#include <iostream>
#include "example-utils.hpp"
#include <csignal>
#include <ctime>
#include <sys/time.h>
#include <unistd.h>
#include <fstream> // Include the header for file operations
#include <chrono>

using namespace std;
unsigned int frameCountperSec = 0;
unsigned int frameCountperMin = 0;
unsigned long long frameCount = 0;
unsigned long long elapsedTime = 0;
double timestamp = 0, lastTimestamp = 0;



int main(int argc, char* argv[]) try {    
    string serial;
    if (!device_with_streams({ RS2_STREAM_COLOR, RS2_STREAM_DEPTH }, serial))
        return EXIT_SUCCESS;

    rs2::colorizer c; // Helper to colorize depth images
    // Create a pipeline to easily configure and start the camera
    rs2::pipeline pipe;
    rs2::config cfg;
    if (!serial.empty())
        cfg.enable_device(serial);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8,60);
    pipe.start(cfg);

    // align object, used to align to depth viewport.
    // Creating align object is an expensive operation
    // that should not be performed in the main loop
    rs2::align align_to_depth(RS2_STREAM_DEPTH);
    rs2::frameset frameset;
    while (1) 
    {
        // rs2::frameset frameset = pipe.wait_for_frames();
        if (pipe.poll_for_frames(&frameset))
        {
            // Align all frames to depth viewport
            auto start = std::chrono::high_resolution_clock::now();
            frameset = align_to_depth.process(frameset);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "align_to_depth Elapsed time: " << duration.count() << " milliseconds" << std::endl;
            // With the aligned frameset, we proceed as usual
            // auto depth = frameset.get_depth_frame();
            // auto color = frameset.get_color_frame();
            rs2::frame color = frameset.get_color_frame();
            rs2::depth_frame depth = frameset.get_depth_frame();

            // auto colorized_depth = c.colorize(depth);

            if (depth && color){    
                auto depth_frame = depth.get_data();
                auto color_frame = color.get_data();
                
                frameCount++;
                frameCountperSec++;
                frameCountperMin++;
            }
        }
    }
    return EXIT_SUCCESS;
} catch (const rs2::error& e) {
    cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << endl;
    return EXIT_FAILURE;
} catch (const exception& e) {
    cerr << e.what() << endl;
    return EXIT_FAILURE;
}
