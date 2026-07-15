from rknnlite.api import RKNNLite
import numpy as np


class RKNN_model_container():
    def __init__(self, model_path, target=None, device_id=None) -> None:
        rknn = RKNNLite()

        # Direct Load RKNN Model
        rknn.load_rknn(model_path)

        print('--> Init runtime environment')
        ret = rknn.init_runtime()
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')
        
        self.rknn = rknn 

    def run(self, inputs):
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            pass
        else:
            inputs = [inputs]
        
        reshaped_inputs = []
        for inp in inputs:
            if inp.ndim == 3:
                #print('--> input has 3 dimensions')
                inp = np.expand_dims(inp, axis=0)
            if inp.ndim != 4:
                raise ValueError(f"Input should have 4 dimensions, but got {inp.ndim} dimensions")
            reshaped_inputs.append(inp)

        result = self.rknn.inference(inputs=reshaped_inputs)
    
        return result