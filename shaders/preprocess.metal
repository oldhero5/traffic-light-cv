#include <metal_stdlib>
using namespace metal;

/*
 * M1 MacBook Pro optimized image preprocessing shaders
 * 
 * Performance targets:
 * - 4K preprocessing: >120 FPS
 * - 1080p preprocessing: >240 FPS
 * - Power consumption: <2W
 * - Memory bandwidth: <50% utilization
 */

// Constants for preprocessing
constant float3 mean_rgb = float3(0.485, 0.456, 0.406);  // ImageNet mean
constant float3 std_rgb = float3(0.229, 0.224, 0.225);   // ImageNet std

/**
 * Resize and normalize image for YOLO inference
 * Optimized for M1 tile-based rendering architecture
 */
kernel void resize_and_normalize(
    texture2d<float, access::read> input_texture [[texture(0)]],
    texture2d<float, access::write> output_texture [[texture(1)]],
    constant uint2& target_size [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Check bounds
    if (gid.x >= target_size.x || gid.y >= target_size.y) {
        return;
    }
    
    // Calculate source coordinates with bilinear sampling
    float2 input_size = float2(input_texture.get_width(), input_texture.get_height());
    float2 scale = input_size / float2(target_size);
    float2 src_coord = (float2(gid) + 0.5) * scale - 0.5;
    
    // Sample input texture with bilinear interpolation
    constexpr sampler linear_sampler(mag_filter::linear, min_filter::linear);
    float4 pixel = input_texture.sample(linear_sampler, src_coord / input_size);
    
    // Convert BGR to RGB and normalize
    float3 rgb = pixel.bgr;  // OpenCV uses BGR, we need RGB
    rgb = (rgb - mean_rgb) / std_rgb;
    
    // Write to output texture
    output_texture.write(float4(rgb, 1.0), gid);
}

/**
 * Fast color space conversion optimized for traffic light detection
 * RGB to HSV conversion for better color discrimination
 */
kernel void rgb_to_hsv(
    texture2d<float, access::read> input_texture [[texture(0)]],
    texture2d<float, access::write> output_texture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= input_texture.get_width() || gid.y >= input_texture.get_height()) {
        return;
    }
    
    float4 rgb_pixel = input_texture.read(gid);
    float3 rgb = rgb_pixel.rgb;
    
    // RGB to HSV conversion
    float max_val = max(max(rgb.r, rgb.g), rgb.b);
    float min_val = min(min(rgb.r, rgb.g), rgb.b);
    float delta = max_val - min_val;
    
    float h = 0.0;
    float s = (max_val > 0.0) ? delta / max_val : 0.0;
    float v = max_val;
    
    if (delta > 0.0) {
        if (max_val == rgb.r) {
            h = (rgb.g - rgb.b) / delta;
        } else if (max_val == rgb.g) {
            h = 2.0 + (rgb.b - rgb.r) / delta;
        } else {
            h = 4.0 + (rgb.r - rgb.g) / delta;
        }
        h *= 60.0;
        if (h < 0.0) h += 360.0;
    }
    
    output_texture.write(float4(h / 360.0, s, v, 1.0), gid);
}

/**
 * Gaussian blur optimized for M1 architecture
 * Uses separable filter for better performance
 */
kernel void gaussian_blur_horizontal(
    texture2d<float, access::read> input_texture [[texture(0)]],
    texture2d<float, access::write> output_texture [[texture(1)]],
    constant float* blur_weights [[buffer(0)]],
    constant int& kernel_size [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output_texture.get_width() || gid.y >= output_texture.get_height()) {
        return;
    }
    
    float4 sum = float4(0.0);
    int half_kernel = kernel_size / 2;
    
    for (int i = -half_kernel; i <= half_kernel; i++) {
        int x = int(gid.x) + i;
        x = clamp(x, 0, int(input_texture.get_width()) - 1);
        
        float4 pixel = input_texture.read(uint2(x, gid.y));
        sum += pixel * blur_weights[i + half_kernel];
    }
    
    output_texture.write(sum, gid);
}

kernel void gaussian_blur_vertical(
    texture2d<float, access::read> input_texture [[texture(0)]],
    texture2d<float, access::write> output_texture [[texture(1)]],
    constant float* blur_weights [[buffer(0)]],
    constant int& kernel_size [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output_texture.get_width() || gid.y >= output_texture.get_height()) {
        return;
    }
    
    float4 sum = float4(0.0);
    int half_kernel = kernel_size / 2;
    
    for (int i = -half_kernel; i <= half_kernel; i++) {
        int y = int(gid.y) + i;
        y = clamp(y, 0, int(input_texture.get_height()) - 1);
        
        float4 pixel = input_texture.read(uint2(gid.x, y));
        sum += pixel * blur_weights[i + half_kernel];
    }
    
    output_texture.write(sum, gid);
}

/**
 * Edge detection for improved traffic light localization
 * Uses Sobel operator optimized for M1
 */
kernel void sobel_edge_detection(
    texture2d<float, access::read> input_texture [[texture(0)]],
    texture2d<float, access::write> output_texture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output_texture.get_width() || gid.y >= output_texture.get_height() ||
        gid.x == 0 || gid.y == 0 ||
        gid.x == input_texture.get_width() - 1 || gid.y == input_texture.get_height() - 1) {
        output_texture.write(float4(0.0), gid);
        return;
    }
    
    // Sample 3x3 neighborhood
    float3x3 neighborhood;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            uint2 coord = uint2(gid.x + i - 1, gid.y + j - 1);
            float4 pixel = input_texture.read(coord);
            neighborhood[i][j] = dot(pixel.rgb, float3(0.299, 0.587, 0.114)); // Grayscale
        }
    }
    
    // Sobel X kernel
    float gx = neighborhood[2][0] + 2.0 * neighborhood[2][1] + neighborhood[2][2] -
               neighborhood[0][0] - 2.0 * neighborhood[0][1] - neighborhood[0][2];
    
    // Sobel Y kernel  
    float gy = neighborhood[0][2] + 2.0 * neighborhood[1][2] + neighborhood[2][2] -
               neighborhood[0][0] - 2.0 * neighborhood[1][0] - neighborhood[2][0];
    
    // Compute magnitude
    float magnitude = sqrt(gx * gx + gy * gy);
    
    output_texture.write(float4(magnitude, magnitude, magnitude, 1.0), gid);
}

/**
 * Histogram equalization for improved contrast
 * Optimized for M1 parallel processing
 */
kernel void histogram_equalization(
    texture2d<float, access::read> input_texture [[texture(0)]],
    texture2d<float, access::write> output_texture [[texture(1)]],
    constant float* cdf [[buffer(0)]],  // Cumulative distribution function
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output_texture.get_width() || gid.y >= output_texture.get_height()) {
        return;
    }
    
    float4 pixel = input_texture.read(gid);
    
    // Apply histogram equalization to each channel
    float3 equalized;
    equalized.r = cdf[int(pixel.r * 255.0)];
    equalized.g = cdf[int(pixel.g * 255.0)]; 
    equalized.b = cdf[int(pixel.b * 255.0)];
    
    output_texture.write(float4(equalized, pixel.a), gid);
}

/**
 * Brightness and contrast adjustment optimized for dashcam footage
 */
kernel void brightness_contrast(
    texture2d<float, access::read> input_texture [[texture(0)]],
    texture2d<float, access::write> output_texture [[texture(1)]],
    constant float& brightness [[buffer(0)]],
    constant float& contrast [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output_texture.get_width() || gid.y >= output_texture.get_height()) {
        return;
    }
    
    float4 pixel = input_texture.read(gid);
    
    // Apply brightness and contrast
    float3 adjusted = (pixel.rgb - 0.5) * contrast + 0.5 + brightness;
    adjusted = clamp(adjusted, 0.0, 1.0);
    
    output_texture.write(float4(adjusted, pixel.a), gid);
}

/**
 * Multi-scale preprocessing for improved detection at various distances
 * Combines multiple resolution outputs
 */
kernel void multi_scale_preprocess(
    texture2d<float, access::read> input_texture [[texture(0)]],
    texture2d<float, access::write> output_scale1 [[texture(1)]],
    texture2d<float, access::write> output_scale2 [[texture(2)]],
    texture2d<float, access::write> output_scale3 [[texture(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Scale 1: Full resolution (for close objects)
    if (gid.x < output_scale1.get_width() && gid.y < output_scale1.get_height()) {
        float2 coord = float2(gid) / float2(output_scale1.get_width(), output_scale1.get_height());
        constexpr sampler linear_sampler(mag_filter::linear, min_filter::linear);
        float4 pixel = input_texture.sample(linear_sampler, coord);
        output_scale1.write(pixel, gid);
    }
    
    // Scale 2: Half resolution (for medium objects)
    if (gid.x < output_scale2.get_width() && gid.y < output_scale2.get_height()) {
        float2 coord = float2(gid) / float2(output_scale2.get_width(), output_scale2.get_height());
        constexpr sampler linear_sampler(mag_filter::linear, min_filter::linear);
        float4 pixel = input_texture.sample(linear_sampler, coord);
        output_scale2.write(pixel, gid);
    }
    
    // Scale 3: Quarter resolution (for distant objects)
    if (gid.x < output_scale3.get_width() && gid.y < output_scale3.get_height()) {
        float2 coord = float2(gid) / float2(output_scale3.get_width(), output_scale3.get_height());
        constexpr sampler linear_sampler(mag_filter::linear, min_filter::linear);
        float4 pixel = input_texture.sample(linear_sampler, coord);
        output_scale3.write(pixel, gid);
    }
}