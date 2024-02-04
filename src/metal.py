import ctypes
import numpy as np

class Metal:
    def __init__(self, tracks):
        self.shader = ctypes.CDLL("./src/shaders/compiled_shader.dylib")
        self.init_argtypes()
        self.init_tracks(tracks)

    def getTrackIndexes(self):
        return self.track_index

    def getPointsOffset(self, input_num):
        self.shader.get_points_offsets(input_num)
    
    def showBuffer(self, count):
        self.shader.show_buffer(count)

    def AddTrackBuffer(self, track_index, track_data):
        track_data = track_data.flatten().astype(np.uint8)
        track_data = track_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        self.shader.add_track(track_index, track_data)
  
        del track_data

    def GetTrackFromBuffer(self, track_index):
        pointer = self.shader.get_track_pointer(track_index)
        numpy_array = np.ctypeslib.as_array(pointer, shape=(5000,5000))
        return numpy_array

    def init_shaders(self, inVectorBufferCount, outVectorBufferCount, offsets):
        pointer = self.shader.init_input_buffer(inVectorBufferCount)
        numpy_array_in = np.ctypeslib.as_array(pointer, shape=(inVectorBufferCount,))
        self.inVectorBuffer = numpy_array_in

        pointer = self.shader.init_output_buffer(outVectorBufferCount)
        numpy_array_out = np.ctypeslib.as_array(pointer, shape=(outVectorBufferCount,))
        self.outVectorBuffer = numpy_array_out

        pointer = self.shader.init_offsets_buffer(len(offsets))
        numpy_array_offsets = np.ctypeslib.as_array(pointer, shape=(len(offsets),))
        self.offsetsBuffer = numpy_array_offsets

        for i in range(len(offsets)):
            self.offsetsBuffer[i] = offsets[i]

    def init_tracks(self, tracks):
        self.track_index = {}    
        increment = 0

        for track_name in tracks.keys():
            self.track_index[track_name] = increment
            self.AddTrackBuffer(increment, tracks[track_name])
            increment += 1

        self.shader.concatenate_tracks()

    def init_argtypes(self):
        self.shader.get_points_offsets.argtypes = [ ctypes.c_int ]
        self.shader.get_track_pointer.argtypes = [ ctypes.c_int ]

        self.shader.add_track.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
        ]
        self.shader.concatenate_tracks.argtypes = []


        self.shader.init_input_buffer.argtypes = [ ctypes.c_int64 ]
        self.shader.init_output_buffer.argtypes = [ ctypes.c_int64 ]
        self.shader.init_offsets_buffer.argtypes = [ ctypes.c_int64 ]

        self.shader.show_buffer.argtypes = [ ctypes.c_int64 ]

        self.shader.init_input_buffer.restype = ctypes.POINTER(ctypes.c_int16)
        self.shader.init_output_buffer.restype = ctypes.POINTER(ctypes.c_int16)
        self.shader.init_offsets_buffer.restype = ctypes.POINTER(ctypes.c_int16)
        self.shader.get_track_pointer.restype = ctypes.POINTER(ctypes.c_uint8)

        