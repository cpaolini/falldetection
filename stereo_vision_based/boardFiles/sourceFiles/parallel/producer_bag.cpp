#include <librealsense2/rs.hpp>
#include "example-utils.hpp"
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <csignal>
#include <ctime>
#include <sys/time.h>
#include <unistd.h>
#include <fstream> 
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <semaphore.h> 
#include <signal.h>

using namespace std;
unsigned int frameCountperSec = 0;
unsigned int frameCountperMin = 0;
unsigned long int frameCount = 0;
unsigned long long elapsedTime = 0;
double timestamp = 0, lastTimestamp = 0;
const char* fifoPath = "myfifo";
sem_t* fifoSemaphore;

const int width = 640;
const int height = 480;
const int numChannels = 3;


// Define a file stream object for logging
// ofstream logFile;

void signal_handler(int signum) {
    cerr << "Received signal: " << signum << " - ";

    switch (signum) {
        case SIGHUP:
            cerr << "Hang up signal (SIGHUP)" << endl;
            break;
        case SIGINT:
            cerr << "Interrupt signal (SIGINT)" << endl;
            break;
        case SIGQUIT:
            cerr << "Quit signal (SIGQUIT)" << endl;
            break;
        case SIGILL:
            cerr << "Illegal instruction signal (SIGILL)" << endl;
            break;
        case SIGTRAP:
            cerr << "Trace or breakpoint trap signal (SIGTRAP)" << endl;
            break;
        case SIGABRT:
        // case SIGIOT:
            cerr << "Abnormal termination signal (SIGABRT/SIGIOT)" << endl;
            break;
        case SIGBUS:
            cerr << "Bus error signal (SIGBUS)" << endl;
            break;
        case SIGFPE:
            cerr << "Floating-Point Exception signal (SIGFPE)" << endl;
            break;
        case SIGKILL:
            cerr << "Kill signal (SIGKILL)" << endl;
            break;
        case SIGUSR1:
            cerr << "User-Defined Signal 1 (SIGUSR1)" << endl;
            break;
        case SIGSEGV:
            cerr << "Segmentation Fault signal (SIGSEGV)" << endl;
            break;
        case SIGUSR2:
            cerr << "User-Defined Signal 2 (SIGUSR2)" << endl;
            break;
        case SIGPIPE:
            cerr << "Broken Pipe signal (SIGPIPE)" << endl;
            break;
        case SIGALRM:
            cerr << "Alarm Clock signal (SIGALRM)" << endl;
            break;
        case SIGTERM:
            cerr << "Terminate signal (SIGTERM)" << endl;
            break;
        case SIGSTKFLT:
            cerr << "Stack Fault signal (SIGSTKFLT)" << endl;
            break;
        case SIGCHLD:
            cerr << "Child Status Changed signal (SIGCHLD)" << endl;
            break;
        case SIGCONT:
            cerr << "Continue signal (SIGCONT)" << endl;
            break;
        case SIGSTOP:
            cerr << "Stop signal (SIGSTOP)" << endl;
            break;
        case SIGTSTP:
            cerr << "Terminal Stop signal (SIGTSTP)" << endl;
            break;
        case SIGTTIN:
            cerr << "Terminal Input for Background Process signal (SIGTTIN)" << endl;
            break;
        case SIGTTOU:
            cerr << "Terminal Output for Background Process signal (SIGTTOU)" << endl;
            break;
        case SIGURG:
            cerr << "Urgent Data Available on a Socket signal (SIGURG)" << endl;
            break;
        case SIGXCPU:
            cerr << "CPU Time Limit Exceeded signal (SIGXCPU)" << endl;
            break;
        case SIGXFSZ:
            cerr << "File Size Limit Exceeded signal (SIGXFSZ)" << endl;
            break;
        case SIGVTALRM:
            cerr << "Virtual Timer Expiration signal (SIGVTALRM)" << endl;
            break;
        case SIGPROF:
            cerr << "Profiling Timer signal (SIGPROF)" << endl;
            break;
        case SIGWINCH:
            cerr << "Window Size Change signal (SIGWINCH)" << endl;
            break;
        case SIGIO:
            cerr << "I/O Possible on a File Descriptor signal (SIGIO)" << endl;
            break;
        // case SIGPOLL:
        //     cerr << "Pollable Event signal (SIGPOLL)" << endl;
        //     break;
        case SIGPWR:
            cerr << "Power Failure signal (SIGPWR)" << endl;
            break;
        case SIGSYS:
            cerr << "Bad System Call signal (SIGSYS)" << endl;
            break;
        // case SIGUNUSED:
        //     cerr << "Unused signal (SIGUNUSED)" << endl;
        //     break;
        default:
            cerr << "Unknown signal" << endl;
            break;
    }
}

int main(int argc, char* argv[]) try {
    if(argc < 2){
        cerr << "Usage: ./producer_bag <path-to-bag-filename>" << endl;
        return EXIT_FAILURE;
    }
    char* bagFileName = argv[1];
    
    fifoSemaphore = sem_open("/myfifo_sem", O_CREAT, 0644, 1);
    printf("Named pipe opening\n");
    mkfifo(fifoPath, 0666);
    int fd = open(fifoPath, O_WRONLY);
    if (fd == -1) {
        perror("open");
        return EXIT_FAILURE;
    }
    int currentSize = fcntl(fd, F_GETPIPE_SZ);
    if (currentSize == -1) {
        perror("fcntl F_GETPIPE_SZ");
        return EXIT_FAILURE;
    }
    std::cout << "Pipe size            : " << currentSize << std::endl;
    int newSize = 2457600;
    if (fcntl(fd, F_SETPIPE_SZ, newSize) == -1) {
        perror("fcntl F_SETPIPE_SZ");
        return EXIT_FAILURE;
    }

    // Get and print the modified pipe size
    int modifiedSize = fcntl(fd, F_GETPIPE_SZ);
    if (modifiedSize == -1) {
        perror("fcntl F_GETPIPE_SZ");
        return EXIT_FAILURE;
    }

    std::cout << "Pipe (modified) size : " << modifiedSize << std::endl;

    printf("Named pipe opened\n");
    
    // Initialize the log file
    // logFile.open("camera_log_POLL_align_core_0123_depth60fps_color60fps_omp_release_1.txt", ios::app); // Append mode
    // if (!logFile.is_open()) {
    //     cerr << "Error: Could not create/open the log file." << endl;
    //     return EXIT_FAILURE;
    // }

    // string serial;
    // if (!device_with_streams({ RS2_STREAM_COLOR, RS2_STREAM_DEPTH }, serial))
    //     return EXIT_SUCCESS;
    
    rs2::colorizer c; // Helper to colorize depth images
    rs2::pipeline pipe;// Create a pipeline to easily configure and start the camera
    rs2::config cfg;
    cfg.enable_device_from_file(bagFileName, false);
    // cfg.enable_device_from_file("20230527_135246.bag", false);
    // cfg.enable_device_from_file("20230526_180734.bag", false);

    // if (!serial.empty())
    //     cfg.enable_device(serial);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8,30);
    pipe.start(cfg);
    auto  device = pipe.get_active_profile().get_device();
    rs2::frameset frameset;
    rs2::playback playback = device.as<rs2::playback>();
    playback.set_real_time(false);
    rs2::align align_to_depth(RS2_STREAM_DEPTH);
    uint8_t flattened_color_frame[width * height * numChannels];
    while (1) // Application still alive?
    {
        // if (pipe.poll_for_frames(&frameset))
        // {
            frameset = pipe.wait_for_frames();
            // Align all frames to depth viewport
            frameset = align_to_depth.process(frameset);

            // With the aligned frameset, we proceed as usual
            rs2::frame color = frameset.get_color_frame();
            rs2::depth_frame depth = frameset.get_depth_frame();

            // auto colorized_depth = c.colorize(depth);

            if (depth && color){    
                auto depth_frame = (const uint8_t*)depth.get_data();
                auto color_frame = (const uint8_t*)color.get_data();

                frameCount++;
                frameCountperSec++;
                frameCountperMin++;
                printf("frameCount = %ld\n", frameCount);
                uint8_t header = 0xFF;
                uint8_t trailer = 0xEE;
                uint8_t frame_buffer[sizeof(header) + (width * height * numChannels) + (width * height * sizeof(uint16_t)) + sizeof(trailer)];
                
                std::memcpy(frame_buffer, &header, sizeof(header));
                // std::memcpy(frame_buffer + sizeof(header), flattened_color_frame, sizeof(flattened_color_frame));
                std::memcpy(frame_buffer + sizeof(header), color_frame, (width * height * numChannels));
                std::memcpy(frame_buffer + sizeof(header) + (width * height * numChannels), depth_frame, (width * height * sizeof(uint16_t)));
                std::memcpy(frame_buffer + sizeof(header) + (width * height * numChannels) + (width * height * sizeof(uint16_t)), &trailer, sizeof(trailer));
                write(fd, frame_buffer, sizeof(frame_buffer));
                fsync(fd);
                sem_post(fifoSemaphore);
                sem_wait(fifoSemaphore);
            }
        // }
    }

    // Close the log file before exiting
    // logFile.close();
    sem_close(fifoSemaphore);
    sem_unlink("/myfifo_sem");
    close(fd);
    return EXIT_SUCCESS;
} catch (const rs2::error& e) {
    cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << endl;
    return EXIT_FAILURE;
} catch (const exception& e) {
    cerr << e.what() << endl;
    return EXIT_FAILURE;
}
