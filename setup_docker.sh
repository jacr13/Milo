#!/bin/bash

# Check if argument is passed
if [ $# -eq 0 ]; then
    echo "Error: Please provide an argument (base, dev, final, or all)"
    exit 1
fi

# Function to build image based on target
build_image() {
    target="$1"
    tag="$2"
    docker build --target "$target" -t candidj0/milo:"$tag" .
    docker push candidj0/milo:"$tag"
}

# Check argument and build accordingly
case "$1" in
    base)
        build_image "base" "base"
        ;;
    dev)
        build_image "dev" "dev"
        ;;
    final)
        build_image "final" "latest"
        ;;
    all)
        build_image "base" "base"
        build_image "dev" "dev"
        build_image "final" "latest"
        ;;
    *)
        echo "Error: Invalid argument. Please use 'base', 'dev', 'final', or 'all'."
        exit 1
        ;;
esac

exit 0