import Metal
import MetalPerformanceShaders
import Accelerate
import Foundation

let metallib =  "\(#file.replacingOccurrences(of: "/PyMetalBridge.swift", with: ""))/Shaders.metallib"

@available(macOS 10.13, *)
let device = MTLCreateSystemDefaultDevice()!,
    commandQueue = device.makeCommandQueue()!,
    defaultLibrary = try! device.makeLibrary(filepath: metallib)

var trackVectorBuffer: MTLBuffer?
var trackVectorBuffers = [Int: MTLBuffer]()
var allTracksBuffer: MTLBuffer?

var commandBuffer : MTLCommandBuffer?
var computeCommandEncoder : MTLComputeCommandEncoder?

@available(macOS 10.13, *)
@_cdecl("add_track")
public func add_track(track: Int, track_data: UnsafePointer<UInt8>) {
    let trackByteLength = 5000 * 5000 * MemoryLayout<UInt8>.size
    let trackBuffer = UnsafeRawPointer (track_data)
    let trackVectorBuffer = device.makeBuffer(bytes: trackBuffer, length: trackByteLength, options: [])
    trackVectorBuffers[track] = trackVectorBuffer
}

@available(macOS 10.13, *)
@_cdecl("concatenate_tracks")
public func concatenate_tracks() {
    allTracksBuffer = device.makeBuffer(length: 5000 * 5000 * MemoryLayout<UInt8>.size * trackVectorBuffers.count, options: [])
    var offset = 0
    for index in 0...(trackVectorBuffers.count - 1) {
        let buffer = trackVectorBuffers[index]
        let content = UnsafeMutableRawPointer(allTracksBuffer!.contents() + offset)
        content.copyMemory(from: buffer!.contents(), byteCount: buffer!.length)
        offset += buffer!.length
        buffer?.setPurgeableState(.empty)
    }
}

@available(macOS 10.13, *)
@_cdecl("get_points_offsets")
public func get_points_offsets(input: UnsafePointer<Int16>, out: UnsafeMutablePointer<Int16>, count: Int) -> Int {
    return computeOffsets(input: input, out: out, count: count)
}

@available(macOS 10.13, *)
@_cdecl("dot_product")
public func dot_product(input: UnsafePointer<Int32>, weights: UnsafePointer<Float32>, out: UnsafeMutablePointer<Float32>, count: Int) -> Int {
    return computeDotProduct(input: input, weights: weights, out: out, count: count)
}

@available(macOS 10.13, *)
public func computeDotProduct(input: UnsafePointer<Int32>, weights: UnsafePointer<Float32>, out: UnsafeMutablePointer<Float32>, count: Int) -> Int {
    do {
        let num_weights = Int(input[0])

        let inputBuffer = UnsafeRawPointer(input)
        let weightsBuffer = UnsafeRawPointer(weights)

        let commandBuffer = commandQueue.makeCommandBuffer()
        let computeCommandEncoder = commandBuffer!.makeComputeCommandEncoder()

        let dotProductFunction = defaultLibrary.makeFunction(name: "dot_product")!
        do {
            let computePipelineState = try device.makeComputePipelineState(function: dotProductFunction)
            computeCommandEncoder!.setComputePipelineState(computePipelineState)
        } catch {
            fatalError("Error creating compute pipeline state: \(error)")
        }


        let inputByteLength = 10 * MemoryLayout<Int32>.size * count + 2
        let weightsByteLength = num_weights * MemoryLayout<Float32>.size
        
        let inVectorBuffer = device.makeBuffer(bytes: inputBuffer, length: inputByteLength, options: [])
        let weightsVectorBuffer = device.makeBuffer(bytes: weightsBuffer, length: weightsByteLength, options: [])

        computeCommandEncoder!.setBuffer(inVectorBuffer, offset: 0, index: 0)
        computeCommandEncoder!.setBuffer(weightsVectorBuffer, offset: 0, index: 1)

        let resultRef = UnsafeMutablePointer<Float32>.allocate(capacity: count)
        let outVectorBuffer = device.makeBuffer(bytes: resultRef, length: count * MemoryLayout<Float32>.size, options: [])

        computeCommandEncoder!.setBuffer(outVectorBuffer, offset: 0, index: 2)

        let threadsPerGroup = MTLSize(width: 1, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: count, height: 1, depth: 1)

        computeCommandEncoder!.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        computeCommandEncoder!.endEncoding()
        commandBuffer!.commit()
        commandBuffer!.waitUntilCompleted()

        // unsafe bitcast and assigin result pointer to output

        out.initialize(from: outVectorBuffer!.contents().assumingMemoryBound(to: Float32.self), count: count)

        resultRef.deallocate()

        inVectorBuffer!.setPurgeableState(.empty)
        weightsVectorBuffer!.setPurgeableState(.empty)
        outVectorBuffer!.setPurgeableState(.empty)

        return 0
    }
}

@available(macOS 10.13, *)
public func computeOffsets(input: UnsafePointer<Int16>, out: UnsafeMutablePointer<Int16>, count: Int) -> Int {
    do {

        let inputBuffer = UnsafeRawPointer(input)

        let commandBuffer = commandQueue.makeCommandBuffer()
        let computeCommandEncoder = commandBuffer!.makeComputeCommandEncoder()
      
        let getPointsFunction = defaultLibrary.makeFunction(name: "points_offsets")!
        let computePipelineState = try device.makeComputePipelineState(function: getPointsFunction)
        computeCommandEncoder!.setComputePipelineState(computePipelineState)

        let inputByteLength = 5 * MemoryLayout<Int16>.size * count

        let inVectorBuffer = device.makeBuffer(bytes: inputBuffer, length: inputByteLength, options: [])

        computeCommandEncoder!.setBuffer(inVectorBuffer, offset: 0, index: 0)
        computeCommandEncoder!.setBuffer(allTracksBuffer, offset: 0, index: 1)

        let resultRef = UnsafeMutablePointer<Int16>.allocate(capacity: count)
        let outVectorBuffer = device.makeBuffer(bytes: resultRef, length: count * MemoryLayout<Int16>.size, options: [])

        computeCommandEncoder!.setBuffer(outVectorBuffer, offset: 0, index: 2)
        
        let threadsPerGroup = MTLSize(width: 1, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: count, height: 1, depth: 1)
        
        computeCommandEncoder!.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        computeCommandEncoder!.endEncoding()
        commandBuffer!.commit()

        commandBuffer!.waitUntilCompleted()

        out.initialize(from: outVectorBuffer!.contents().assumingMemoryBound(to: Int16.self), count: count)

        resultRef.deallocate()

        inVectorBuffer!.setPurgeableState(.empty)
        outVectorBuffer!.setPurgeableState(.empty)
        
        return 0 
    } catch {
        print("\(error)")
        return 1
    }
}