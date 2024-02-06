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
var inBuffer: MTLBuffer?
var outBuffer: MTLBuffer?
var offsetsBuffer: MTLBuffer?

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
@_cdecl("get_track_pointer")
public func get_track_pointer(track: Int) -> UnsafeMutablePointer<UInt8> {
    let offsetInBytes = track * 5000 * 5000 * MemoryLayout<UInt8>.size
    let pointer = allTracksBuffer?.contents().advanced(by: offsetInBytes).bindMemory(to: UInt8.self, capacity: 5000 * 5000)
    return pointer!
}

@available(macOS 10.13, *)
@_cdecl("get_points_offsets")
public func get_points_offsets(count: Int) -> Int {
    return computeOffsets(count: count)
}


@available(macOS 10.13, *)
public func computeOffsets(count: Int) -> Int {
    do {
        let commandBuffer = commandQueue.makeCommandBuffer()
        let computeCommandEncoder = commandBuffer!.makeComputeCommandEncoder()
      
        let getPointsFunction = defaultLibrary.makeFunction(name: "points_offsets")!
        let computePipelineState = try device.makeComputePipelineState(function: getPointsFunction)
        computeCommandEncoder!.setComputePipelineState(computePipelineState)

        computeCommandEncoder!.setBuffer(inBuffer, offset: 0, index: 0)
        computeCommandEncoder!.setBuffer(allTracksBuffer, offset: 0, index: 1)
        computeCommandEncoder!.setBuffer(outBuffer, offset: 0, index: 2)
        computeCommandEncoder!.setBuffer(offsetsBuffer, offset: 0, index: 3)
        
        let threadsPerGroup = MTLSize(width: 1, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: count, height: 1, depth: 1)
        
        computeCommandEncoder!.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        computeCommandEncoder!.endEncoding()
        commandBuffer!.commit()
        commandBuffer!.waitUntilCompleted()
        
        return 0 
    } catch {
        print("\(error)")
        return 1
    }
}

@available(macOS 10.13, *)
@_cdecl("init_input_buffer")
public func init_input_buffer(count: Int) -> UnsafeMutablePointer<Int16> {
    let bufferSize = MemoryLayout<Int16>.size * count
    let inVectorBuffer = device.makeBuffer(length: bufferSize, options: [])!

    let pointer = inVectorBuffer.contents().bindMemory(to: Int16.self, capacity: count)

    if inBuffer != nil { inBuffer!.setPurgeableState(.empty) }
    inBuffer = inVectorBuffer

    return pointer
}

@available(macOS 10.13, *)
@_cdecl("init_output_buffer")
public func init_output_buffer(count: Int) -> UnsafeMutablePointer<Int16> {
    let bufferSize = MemoryLayout<Int16>.size * count
    let outVectorBuffer = device.makeBuffer(length: bufferSize, options: [])!

    let pointer = outVectorBuffer.contents().bindMemory(to: Int16.self, capacity: count)
    
    if outBuffer != nil { outBuffer!.setPurgeableState(.empty) }
    outBuffer = outVectorBuffer

    return pointer
}

@available(macOS 10.13, *)
@_cdecl("init_offsets_buffer")
public func init_offsets_buffer(count: Int) -> UnsafeMutablePointer<Int16> {
    let bufferSize = MemoryLayout<Int16>.size * count
    let offsetsVectorBuffer = device.makeBuffer(length: bufferSize, options: [])!

    let pointer = offsetsVectorBuffer.contents().bindMemory(to: Int16.self, capacity: count)

    if offsetsBuffer != nil { offsetsBuffer!.setPurgeableState(.empty) }
    offsetsBuffer = offsetsVectorBuffer

    return pointer
}

@available(macOS 10.13, *)
@_cdecl("show_buffer")
public func show_buffer(count: Int) {
    let pointer = offsetsBuffer!.contents().bindMemory(to: Int16.self, capacity: count)
    for i in 0...count { print(pointer[i], terminator: " ") }
    print()
}
