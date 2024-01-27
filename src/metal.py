import ctypes
import numpy as np

class Metal:
    def __init__(self, tracks):
        self.shader = ctypes.CDLL("./src/shaders/compiled_shader.dylib")
        self.init_argtypes()
        self.init_tracks(tracks)

    def getTrackIndexes(self):
        return self.track_index

    def getPointsOffset(self, input_data):
        input_ptr = (input_data.astype(np.short)).ctypes.data_as(ctypes.POINTER(ctypes.c_short))
        output_mutable_ptr = (ctypes.c_short * (int(len(input_data)/5)))()
        self.shader.get_points_offsets(input_ptr, output_mutable_ptr, int(len(input_data)/5))
        output = np.array(output_mutable_ptr)
        del input_ptr    
        del output_mutable_ptr
        return output
    
    def dot_product(self, input_info, input_data, products):
        input_info = np.array(input_info, dtype=np.int32)
        input_data = np.array(input_data, dtype=np.float32)

        input_info_ptr = input_info.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        input_data_ptr = input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        output = np.zeros(products, dtype=np.float32)
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self.shader.dot_product(input_info_ptr, input_data_ptr, output_ptr, products)

        return output

    def AddTrackBuffer(self, track_index, track_data):
        track_data = track_data.flatten().astype(np.uint8)
        track_data = track_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        self.shader.add_track(track_index, track_data)
  
        del track_data

    def init_tracks(self, tracks):
        self.track_index = {}    
        increment = 0

        for track_name in tracks.keys():
            self.track_index[track_name] = increment
            self.AddTrackBuffer(increment, tracks[track_name])
            increment += 1

        self.shader.concatenate_tracks()

    def init_argtypes(self):
        self.shader.get_points_offsets.argtypes = [
            ctypes.POINTER(ctypes.c_short),
            ctypes.POINTER(ctypes.c_short), 
            ctypes.c_int
        ]

        self.shader.add_track.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
        ]

        self.shader.dot_product.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]


        self.shader.concatenate_tracks.argtypes = []