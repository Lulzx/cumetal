#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dispatch/dispatch.h>
#include <stdio.h>

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: %s <path-to-metallib>\n", argv[0]);
            return 64;
        }

        NSString* path = [NSString stringWithUTF8String:argv[1]];
        NSError* read_error = nil;
        NSData* data = [NSData dataWithContentsOfFile:path options:0 error:&read_error];
        if (data == nil || data.length == 0) {
            if (read_error != nil) {
                fprintf(stderr, "SKIP: failed to read metallib: %s\n",
                        read_error.localizedDescription.UTF8String);
            } else {
                fprintf(stderr, "SKIP: metallib file missing or empty at %s\n", argv[1]);
            }
            return 77;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            fprintf(stderr, "SKIP: no Metal device available\n");
            return 77;
        }

        dispatch_data_t dispatch_data = dispatch_data_create(
            data.bytes, data.length, dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT);

        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithData:dispatch_data error:&error];

        if (library == nil || error != nil) {
            fprintf(stderr, "FAIL: newLibraryWithData error: %s\n",
                    error == nil ? "unknown" : error.localizedDescription.UTF8String);
            return 1;
        }

        printf("PASS: loaded metallib with %lu bytes\n", (unsigned long)data.length);
        return 0;
    }
}
