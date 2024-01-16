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

@available(macOS 10.13, *)
@_cdecl("get_points_offsets")
public func get_points_offsets(track_id : Int, input: UnsafePointer<Int32>, out: UnsafeMutablePointer<Int32>, count: Int) -> Int {
    return computeOffsets(track_id : track_id, input: input, out: out, count: count)
}

@available(macOS 10.13, *)
@_cdecl("add_track")
public func add_track(track: Int, track_data: UnsafePointer<Int32>) {
    let trackByteLength = 5000 * 5000 * MemoryLayout<Int32>.size
    let trackBuffer = UnsafeRawPointer (track_data)
    let trackVectorBuffer = device.makeBuffer(bytes: trackBuffer, length: trackByteLength, options: [])
    trackVectorBuffers[track] = trackVectorBuffer
}

@available(macOS 10.13, *)
@_cdecl("concatenate_tracks")
public func concatenate_tracks() {
    let trackByteLength = 5000 * 5000 * MemoryLayout<Int32>.size * trackVectorBuffers.count
    let allTracksBuffer = device.makeBuffer(length: trackByteLength, options: [])
    var offset = 0
    for index in 0...(trackVectorBuffers.count - 1) {
        let buffer = trackVectorBuffers[index]
        let content = UnsafeMutableRawPointer(allTracksBuffer!.contents() + offset)
        content.copyMemory(from: buffer!.contents(), byteCount: buffer!.length)
        offset += buffer!.length
    }
}

@available(macOS 10.13, *)
public func computeOffsets(track_id: Int, input: UnsafePointer<Int32>, out: UnsafeMutablePointer<Int32>, count: Int) -> Int {
    do {
        let inputBuffer = UnsafeRawPointer(input)
        let trackVectorBuffer = trackVectorBuffers[track_id]
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!

        let getPointsFunction = defaultLibrary.makeFunction(name: "points_offsets")!
        let computePipelineState = try device.makeComputePipelineState(function: getPointsFunction)
        computeCommandEncoder.setComputePipelineState(computePipelineState)

        let inputByteLength = 4 * MemoryLayout<Int32>.size * count

        let inVectorBuffer = device.makeBuffer(bytes: inputBuffer, length: inputByteLength, options: [])

        computeCommandEncoder.setBuffer(inVectorBuffer, offset: 0, index: 0)
        computeCommandEncoder.setBuffer(trackVectorBuffer, offset: 0, index: 1)

        let resultRef = UnsafeMutablePointer<Int32>.allocate(capacity: count * 2)
        let outVectorBuffer = device.makeBuffer(bytes: resultRef, length: count * 2 * MemoryLayout<Int32>.size, options: [])

        computeCommandEncoder.setBuffer(outVectorBuffer, offset: 0, index: 2)
        
        let threadsPerGroup = MTLSize(width: 1, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: count, height: 1, depth: 1)

        computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        computeCommandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // unsafe bitcast and assigin result pointer to output

        out.initialize(from: outVectorBuffer!.contents().assumingMemoryBound(to: Int32.self), count: count * 2)

        resultRef.deallocate()

        inVectorBuffer!.setPurgeableState(.empty)
        outVectorBuffer!.setPurgeableState(.empty)
        
        
        return 0 
    } catch {
        print("\(error)")
        return 1
    }
}