#!/bin/bash

# Create shaders directory if it doesn't exist
mkdir -p shaders

# Compile vertex shader
glslc shaders/triangle.vert -o shaders/vert.spv

# Compile fragment shader
glslc shaders/triangle.frag -o shaders/frag.spv

echo "Shaders compiled successfully!"