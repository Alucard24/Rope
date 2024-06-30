import cv2
import numpy as np
from skimage import transform as trans
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from numpy.linalg import norm as l2norm
import onnxruntime
import onnx
from itertools import product as product
import subprocess as sp
onnxruntime.set_default_logger_severity(4)
onnxruntime.log_verbosity_level = -1
import rope.FaceUtil as faceutil
import pickle

class Models():
    def __init__(self):
        self.arcface_dst = np.array( [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
        # self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.providers = [
            ('TensorrtExecutionProvider', {
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': "tensorrt-engines",
                'trt_timing_cache_enable': True,
                'trt_timing_cache_path': "tensorrt-engines",
                'trt_dump_ep_context_model': True,
                'trt_ep_context_file_path': "tensorrt-engines",
            }),
            ('CUDAExecutionProvider'),
            ('CPUExecutionProvider')]
        self.retinaface_model = []
        self.yoloface_model = []
        self.scrdf_model = []
        self.yunet_model = []
        self.face_landmark_68_model = []
        self.face_landmark_3d68_model = []
        self.mean_lmk = []
        self.face_landmark_98_model = []
        self.face_landmark_106_model = []
        self.face_landmark_478_model = []
        self.face_blendshapes_model = []
        self.resnet50_model, self.anchors  = [], []

        self.insight106_model = []

        self.recognition_model = []
        self.recognition_simswap_model = []
        self.recognition_ghost_model = []
        self.swapper_model = []
        self.swapper_model_kps = []
        self.swapper_model_swap = []
        self.simswap512_model = []
        self.ghostfacev1swap_model = []
        self.ghostfacev2swap_model = []
        self.ghostfacev3swap_model = []

        self.emap = []
        self.GFPGAN_model = []
        self.GPEN_256_model = []
        self.GPEN_512_model = []
        self.GPEN_1024_model = []
        self.GPEN_2048_model = []
        self.codeformer_model = []

        self.occluder_model = []
        self.faceparser_model = []

        self.syncvec = torch.empty((1,1), dtype=torch.float32, device='cuda:0')

        self.normalize = v2.Normalize(mean = [ 0., 0., 0. ],
                                      std = [ 1/1.0, 1/1.0, 1/1.0 ])

        self.LandmarksSubsetIdxs = [
            0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39,
            40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
            81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133,
            136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160,
            161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246,
            249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295,
            296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334,
            336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
            384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
            466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477
        ]

    def get_gpu_memory(self):
        command = "nvidia-smi --query-gpu=memory.total --format=csv"
        memory_total_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_total = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]

        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

        memory_used = memory_total[0] - memory_free[0]

        return memory_used, memory_total[0]

    def run_detect(self, img, detect_mode='Retinaface', max_num=1, score=0.5, use_landmark_detection=False, landmark_detect_mode='98', landmark_score=0.5, from_points=False):
        bboxes = []
        kpss = []

        if detect_mode=='Retinaface':
            if not self.retinaface_model:
                self.retinaface_model = onnxruntime.InferenceSession('./models/det_10g.onnx', providers=self.providers)

            bboxes, kpss = self.detect_retinaface(img, max_num=max_num, score=score, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points)

        elif detect_mode=='SCRDF':
            if not self.scrdf_model:
                self.scrdf_model = onnxruntime.InferenceSession('./models/scrfd_2.5g_bnkps.onnx', providers=self.providers)

            bboxes, kpss = self.detect_scrdf(img, max_num=max_num, score=score, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points)

        elif detect_mode=='Yolov8':
            if not self.yoloface_model:
                self.yoloface_model = onnxruntime.InferenceSession('./models/yoloface_8n.onnx', providers=self.providers)
                #self.insight106_model = onnxruntime.InferenceSession('./models/2d106det.onnx', providers=self.providers)

            bboxes, kpss = self.detect_yoloface(img, max_num=max_num, score=score, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points)

        elif detect_mode=='Yunet':
            if not self.yunet_model:
                self.yunet_model = onnxruntime.InferenceSession('./models/yunet_n_640_640.onnx', providers=self.providers)

            bboxes, kpss = self.detect_yunet(img, max_num=max_num, score=score, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points)

        return bboxes, kpss

    def run_detect_landmark(self, img, bbox, det_kpss, detect_mode='98', score=0.5, from_points=False):
        kpss = []
        scores = []

        if detect_mode=='5':
            if not self.resnet50_model:
                self.resnet50_model = onnxruntime.InferenceSession("./models/res50.onnx", providers=self.providers)

                feature_maps = [[64, 64], [32, 32], [16, 16]]
                min_sizes = [[16, 32], [64, 128], [256, 512]]
                steps = [8, 16, 32]
                image_size = 512

                for k, f in enumerate(feature_maps):
                    min_size_array = min_sizes[k]
                    for i, j in product(range(f[0]), range(f[1])):
                        for min_size in min_size_array:
                            s_kx = min_size / image_size
                            s_ky = min_size / image_size
                            dense_cx = [x * steps[k] / image_size for x in [j + 0.5]]
                            dense_cy = [y * steps[k] / image_size for y in [i + 0.5]]
                            for cy, cx in product(dense_cy, dense_cx):
                                self.anchors += [cx, cy, s_kx, s_ky]

            kpss, scores = self.detect_face_landmark_5(img, bbox=bbox, det_kpss=det_kpss, from_points=from_points)

        elif detect_mode=='68':
            if not self.face_landmark_68_model:
                self.face_landmark_68_model = onnxruntime.InferenceSession('./models/2dfan4.onnx', providers=self.providers)

            kpss, scores = self.detect_face_landmark_68(img, bbox=bbox, det_kpss=det_kpss, convert68_5=True, from_points=from_points)

        elif detect_mode=='3d68':
            if not self.face_landmark_3d68_model:
                self.face_landmark_3d68_model = onnxruntime.InferenceSession('./models/1k3d68.onnx', providers=self.providers)
                with open('./models/meanshape_68.pkl', 'rb') as f:
                    self.mean_lmk = pickle.load(f)

            kpss, scores = self.detect_face_landmark_3d68(img, bbox=bbox, det_kpss=det_kpss, convert68_5=True, from_points=from_points)

            return kpss, scores

        elif detect_mode=='98':
            if not self.face_landmark_98_model:
                self.face_landmark_98_model = onnxruntime.InferenceSession('./models/peppapig_teacher_Nx3x256x256.onnx', providers=self.providers)

            kpss, scores = self.detect_face_landmark_98(img, bbox=bbox, det_kpss=det_kpss, convert98_5=True, from_points=from_points)

        elif detect_mode=='106':
            if not self.face_landmark_106_model:
                self.face_landmark_106_model = onnxruntime.InferenceSession('./models/2d106det.onnx', providers=self.providers)

            kpss, scores = self.detect_face_landmark_106(img, bbox=bbox, det_kpss=det_kpss, convert106_5=True, from_points=from_points)

            return kpss, scores

        elif detect_mode=='478':
            if not self.face_landmark_478_model:
                self.face_landmark_478_model = onnxruntime.InferenceSession('./models/face_landmarks_detector_Nx3x256x256.onnx', providers=self.providers)

            if not self.face_blendshapes_model:
                self.face_blendshapes_model = onnxruntime.InferenceSession('./models/face_blendshapes_Nx146x2.onnx', providers=self.providers)

            kpss, scores = self.detect_face_landmark_478(img, bbox=bbox, det_kpss=det_kpss, convert478_5=True, from_points=from_points)

            return kpss, scores

        if len(kpss) > 0:
            if len(scores) > 0:
                if np.mean(scores) >= score:
                    return kpss, scores
            else:
                return kpss, scores

        return [], []

    def delete_models(self):
        self.retinaface_model = []
        self.yoloface_model = []
        self.scrdf_model = []
        self.yunet_model = []
        self.face_landmark_68_model = []
        self.face_landmark_3d68_model = []
        self.mean_lmk = []
        self.face_landmark_98_model = []
        self.face_landmark_106_model = []
        self.face_landmark_478_model = []
        self.face_blendshapes_model = []
        self.resnet50_model = []
        self.insight106_model = []
        self.recognition_model = []
        self.recognition_simswap_model = []
        self.recognition_ghost_model = []
        self.swapper_model = []
        self.simswap512_model = []
        self.ghostfacev1swap_model = []
        self.ghostfacev2swap_model = []
        self.ghostfacev3swap_model = []
        self.GFPGAN_model = []
        self.GPEN_256_model = []
        self.GPEN_512_model = []
        self.GPEN_1024_model = []
        self.GPEN_2048_model = []
        self.codeformer_model = []
        self.occluder_model = []
        self.faceparser_model = []

    def run_recognize(self, img, kps, similarity_type='Opal', face_swapper_model='Inswapper128'):
        if face_swapper_model == 'Inswapper128':
            if not self.recognition_model:
                self.recognition_model = onnxruntime.InferenceSession('./models/w600k_r50.onnx', providers=self.providers)

            embedding, cropped_image = self.recognize(self.recognition_model, img, kps, similarity_type=similarity_type)
        elif face_swapper_model == 'SimSwap512':
            if not self.recognition_simswap_model:
                self.recognition_simswap_model = onnxruntime.InferenceSession('./models/simswap_arcface_model.onnx', providers=self.providers)

            embedding, cropped_image = self.recognize(self.recognition_simswap_model, img, kps, similarity_type=similarity_type)

        elif face_swapper_model == 'GF1' or face_swapper_model == 'GF2' or face_swapper_model == 'GF3':
            if not self.recognition_ghost_model:
                self.recognition_ghost_model = onnxruntime.InferenceSession('./models/ghost_arcface_backbone.onnx', providers=self.providers)

            embedding, cropped_image = self.recognize(self.recognition_ghost_model, img, kps, similarity_type=similarity_type)

        return embedding, cropped_image

    def calc_swapper_latent(self, source_embedding):
        if not self.swapper_model:
            graph = onnx.load("./models/inswapper_128.fp16.onnx").graph
            self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])

        n_e = source_embedding / l2norm(source_embedding)
        latent = n_e.reshape((1,-1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        return latent

    def run_swapper(self, image, embedding, output):
        if not self.swapper_model:
            cuda_options = {"arena_extend_strategy": "kSameAsRequested", 'cudnn_conv_algo_search': 'DEFAULT'}
            sess_options = onnxruntime.SessionOptions()
            sess_options.enable_cpu_mem_arena = False

            # self.swapper_model = onnxruntime.InferenceSession( "./models/inswapper_128_last_cubic.onnx", sess_options, providers=[('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider'])

            self.swapper_model = onnxruntime.InferenceSession( "./models/inswapper_128.fp16.onnx", providers=self.providers)

        io_binding = self.swapper_model.io_binding()
        io_binding.bind_input(name='target', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,128,128), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='source', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,128,128), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.swapper_model.run_with_iobinding(io_binding)

    def calc_swapper_latent_simswap512(self, source_embedding):
        latent = source_embedding.reshape(1, -1)
        latent /= np.linalg.norm(latent)

        return latent

    def run_swapper_simswap512(self, image, embedding, output):
        if not self.simswap512_model:
            self.simswap512_model = onnxruntime.InferenceSession( "./models/simswap_512_unoff.onnx", providers=self.providers)
        
        io_binding = self.simswap512_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='onnx::Gemm_1', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())
        
        self.syncvec.cpu()
        self.simswap512_model.run_with_iobinding(io_binding)

    def calc_swapper_latent_ghost(self, source_embedding):
        latent = source_embedding.reshape((1,-1))

        return latent

    def run_swapper_ghostface(self, image, embedding, output, swapper_model='GF2'):
        if swapper_model == 'GF1':
            if not self.ghostfacev1swap_model:
                self.ghostfacev1swap_model = onnxruntime.InferenceSession( "./models/ghost_unet_1_block.onnx", providers=self.providers)

            ghostfaceswap_model = self.ghostfacev1swap_model
            output_name = '781'
            #output_name2 = 'onnx::ConvTranspose_239'
        elif swapper_model == 'GF2':
            if not self.ghostfacev2swap_model:
                self.ghostfacev2swap_model = onnxruntime.InferenceSession( "./models/ghost_unet_2_block.onnx", providers=self.providers)

            ghostfaceswap_model = self.ghostfacev2swap_model
            output_name = '1165'
            #output_name2 = 'onnx::ConvTranspose_327'
        elif swapper_model == 'GF3':
            if not self.ghostfacev3swap_model:
                self.ghostfacev3swap_model = onnxruntime.InferenceSession( "./models/ghost_unet_3_block.onnx", providers=self.providers)

            ghostfaceswap_model = self.ghostfacev3swap_model
            output_name = '1549'
            #output_name2 = 'onnx::ConvTranspose_415'

        '''
        ouput2 = torch.empty((1,1024,2,2), dtype=torch.float32, device='cuda').contiguous()
        input63 = torch.empty((1,2048,4,4), dtype=torch.float32, device='cuda').contiguous()
        input75 = torch.empty((1,1024,8,8), dtype=torch.float32, device='cuda').contiguous()
        input87 = torch.empty((1,512,16,16), dtype=torch.float32, device='cuda').contiguous()
        input99 = torch.empty((1,256,32,32), dtype=torch.float32, device='cuda').contiguous()
        input111 = torch.empty((1,128,64,64), dtype=torch.float32, device='cuda').contiguous()
        input123 = torch.empty((1,64,128,128), dtype=torch.float32, device='cuda').contiguous()
        input127 = torch.empty((1,64,256,256), dtype=torch.float32, device='cuda').contiguous()
        '''
        
        io_binding = ghostfaceswap_model.io_binding()
        io_binding.bind_input(name='target', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='source', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name=output_name, device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=output.data_ptr())
        '''
        io_binding.bind_output(name=output_name2, device_type='cuda', device_id=0, element_type=np.float32, shape=(1,1024,2,2), buffer_ptr=ouput2.data_ptr())
        io_binding.bind_output(name='input.63', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,2048,4,4), buffer_ptr=input63.data_ptr())
        io_binding.bind_output(name='input.75', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,1024,8,8), buffer_ptr=input75.data_ptr())
        io_binding.bind_output(name='input.87', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,512,16,16), buffer_ptr=input87.data_ptr())
        io_binding.bind_output(name='input.99', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,256,32,32), buffer_ptr=input99.data_ptr())
        io_binding.bind_output(name='input.111', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,128,64,64), buffer_ptr=input111.data_ptr())
        io_binding.bind_output(name='input.123', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,64,128,128), buffer_ptr=input123.data_ptr())
        io_binding.bind_output(name='input.127', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,64,256,256), buffer_ptr=input127.data_ptr())
        '''

        self.syncvec.cpu()
        ghostfaceswap_model.run_with_iobinding(io_binding)

    def run_swap_stg1(self, embedding):

        # Load model
        if not self.swapper_model_kps:
            self.swapper_model_kps = onnxruntime.InferenceSession( "./models/inswapper_kps.onnx", providers=self.providers)

        # Wacky data structure
        io_binding = self.swapper_model_kps.io_binding()
        kps_1 = torch.ones((1, 2048), dtype=torch.float16, device='cuda').contiguous()
        kps_2 = torch.ones((1, 2048), dtype=torch.float16, device='cuda').contiguous()
        kps_3 = torch.ones((1, 2048), dtype=torch.float16, device='cuda').contiguous()
        kps_4 = torch.ones((1, 2048), dtype=torch.float16, device='cuda').contiguous()
        kps_5 = torch.ones((1, 2048), dtype=torch.float16, device='cuda').contiguous()
        kps_6 = torch.ones((1, 2048), dtype=torch.float16, device='cuda').contiguous()
        kps_7 = torch.ones((1, 2048), dtype=torch.float16, device='cuda').contiguous()
        kps_8 = torch.ones((1, 2048), dtype=torch.float16, device='cuda').contiguous()
        kps_9 = torch.ones((1, 2048), dtype=torch.float16, device='cuda').contiguous()
        kps_10 = torch.ones((1, 2048), dtype=torch.float16, device='cuda').contiguous()
        kps_11 = torch.ones((1, 2048), dtype=torch.float16, device='cuda').contiguous()
        kps_12 = torch.ones((1, 2048), dtype=torch.float16, device='cuda').contiguous()

        # Bind the data structures
        io_binding.bind_input(name='source', device_type='cuda', device_id=0, element_type=np.float32, shape=(1, 512), buffer_ptr=embedding.data_ptr())
        io_binding.bind_output(name='1', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=kps_1.data_ptr())
        io_binding.bind_output(name='2', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=kps_2.data_ptr())
        io_binding.bind_output(name='3', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=kps_3.data_ptr())
        io_binding.bind_output(name='4', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=kps_4.data_ptr())
        io_binding.bind_output(name='5', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=kps_5.data_ptr())
        io_binding.bind_output(name='6', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=kps_6.data_ptr())
        io_binding.bind_output(name='7', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=kps_7.data_ptr())
        io_binding.bind_output(name='8', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=kps_8.data_ptr())
        io_binding.bind_output(name='9', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=kps_9.data_ptr())
        io_binding.bind_output(name='10', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=kps_10.data_ptr())
        io_binding.bind_output(name='11', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=kps_11.data_ptr())
        io_binding.bind_output(name='12', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=kps_12.data_ptr())

        self.syncvec.cpu()
        self.swapper_model_kps.run_with_iobinding(io_binding)

        # List of pointers
        holder = []
        holder.append(kps_1)
        holder.append(kps_2)
        holder.append(kps_3)
        holder.append(kps_4)
        holder.append(kps_5)
        holder.append(kps_6)
        holder.append(kps_7)
        holder.append(kps_8)
        holder.append(kps_9)
        holder.append(kps_10)
        holder.append(kps_11)
        holder.append(kps_12)

        return holder


    def run_swap_stg2(self, image, holder, output):
        if not self.swapper_model_swap:
            self.swapper_model_swap = onnxruntime.InferenceSession( "./models/inswapper_swap.onnx", providers=self.providers)

        io_binding = self.swapper_model_swap.io_binding()
        io_binding.bind_input(name='target', device_type='cuda', device_id=0, element_type=np.float32, shape=(1, 3, 128, 128), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='onnx::Unsqueeze_170', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=holder[0].data_ptr())
        io_binding.bind_input(name='onnx::Unsqueeze_224', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=holder[1].data_ptr())
        io_binding.bind_input(name='onnx::Unsqueeze_278', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=holder[2].data_ptr())
        io_binding.bind_input(name='onnx::Unsqueeze_332', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=holder[3].data_ptr())
        io_binding.bind_input(name='onnx::Unsqueeze_386', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=holder[4].data_ptr())
        io_binding.bind_input(name='onnx::Unsqueeze_440', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=holder[5].data_ptr())
        io_binding.bind_input(name='onnx::Unsqueeze_494', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=holder[6].data_ptr())
        io_binding.bind_input(name='onnx::Unsqueeze_548', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=holder[7].data_ptr())
        io_binding.bind_input(name='onnx::Unsqueeze_602', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=holder[8].data_ptr())
        io_binding.bind_input(name='onnx::Unsqueeze_656', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=holder[9].data_ptr())
        io_binding.bind_input(name='onnx::Unsqueeze_710', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=holder[10].data_ptr())
        io_binding.bind_input(name='onnx::Unsqueeze_764', device_type='cuda', device_id=0, element_type=np.float16, shape=(1, 2048), buffer_ptr=holder[11].data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1, 3, 128, 128), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.swapper_model_swap.run_with_iobinding(io_binding)
    def run_GFPGAN(self, image, output):
        if not self.GFPGAN_model:
            # cuda_options = {"arena_extend_strategy": "kSameAsRequested", 'cudnn_conv_algo_search': 'DEFAULT'}
            # sess_options = onnxruntime.SessionOptions()
            # sess_options.enable_cpu_mem_arena = False

            # self.GFPGAN_model = onnxruntime.InferenceSession( "./models/GFPGANv1.4.onnx", sess_options, providers=[("CUDAExecutionProvider", cuda_options), 'CPUExecutionProvider'])

            self.GFPGAN_model = onnxruntime.InferenceSession( "./models/GFPGANv1.4.onnx", providers=self.providers)

        io_binding = self.GFPGAN_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.GFPGAN_model.run_with_iobinding(io_binding)

    def run_GPEN_2048(self, image, output):
        if not self.GPEN_2048_model:
            self.GPEN_2048_model = onnxruntime.InferenceSession( "./models/GPEN-BFR-2048.onnx", providers=self.providers)
 
        io_binding = self.GPEN_2048_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,2048,2048), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,2048,2048), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.GPEN_2048_model.run_with_iobinding(io_binding)

    def run_GPEN_1024(self, image, output):
        if not self.GPEN_1024_model:
            self.GPEN_1024_model = onnxruntime.InferenceSession( "./models/GPEN-BFR-1024.onnx", providers=self.providers)
 
        io_binding = self.GPEN_1024_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,1024,1024), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,1024,1024), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.GPEN_1024_model.run_with_iobinding(io_binding)


    def run_GPEN_512(self, image, output):
        if not self.GPEN_512_model:
            self.GPEN_512_model = onnxruntime.InferenceSession( "./models/GPEN-BFR-512.onnx", providers=self.providers)

        io_binding = self.GPEN_512_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.GPEN_512_model.run_with_iobinding(io_binding)

    def run_GPEN_256(self, image, output):
        if not self.GPEN_256_model:
            self.GPEN_256_model = onnxruntime.InferenceSession( "./models/GPEN-BFR-256.onnx", providers=self.providers)

        io_binding = self.GPEN_256_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.GPEN_256_model.run_with_iobinding(io_binding)

    def run_codeformer(self, image, output):
        if not self.codeformer_model:
            self.codeformer_model = onnxruntime.InferenceSession( "./models/codeformer_fp16.onnx", providers=self.providers)

        io_binding = self.codeformer_model.io_binding()
        io_binding.bind_input(name='x', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        w = np.array([0.9], dtype=np.double)
        io_binding.bind_cpu_input('w', w)
        io_binding.bind_output(name='y', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        self.syncvec.cpu()
        self.codeformer_model.run_with_iobinding(io_binding)

    def run_occluder(self, image, output):
        if not self.occluder_model:
            self.occluder_model = onnxruntime.InferenceSession("./models/occluder.onnx", providers=self.providers)

        io_binding = self.occluder_model.io_binding()
        io_binding.bind_input(name='img', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,1,256,256), buffer_ptr=output.data_ptr())

        # torch.cuda.synchronize('cuda')
        self.syncvec.cpu()
        self.occluder_model.run_with_iobinding(io_binding)

    def run_faceparser(self, image, output):
        if not self.faceparser_model:
            self.faceparser_model = onnxruntime.InferenceSession("./models/faceparser_fp16.onnx", providers=self.providers)

        io_binding = self.faceparser_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='out', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,19,512,512), buffer_ptr=output.data_ptr())

        # torch.cuda.synchronize('cuda')
        self.syncvec.cpu()
        self.faceparser_model.run_with_iobinding(io_binding)

    def detect_retinaface(self, img, max_num, score, use_landmark_detection, landmark_detect_mode, landmark_score, from_points):
        if use_landmark_detection:
            img_landmark = img.clone()

        # Resize image to fit within the input_size
        input_size = (640, 640)
        im_ratio = torch.div(img.size()[1], img.size()[2])

        # model_ratio = float(input_size[1]) / input_size[0]
        model_ratio = 1.0
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = torch.div(new_height,  img.size()[1])

        resize = v2.Resize((new_height, new_width), antialias=True)
        img = resize(img)
        img = img.permute(1,2,0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.float32, device='cuda:0')
        det_img[:new_height,:new_width,  :] = img

        # Switch to BGR and normalize
        det_img = det_img[:, :, [2,1,0]]
        det_img = torch.sub(det_img, 127.5)
        det_img = torch.div(det_img, 128.0)
        det_img = det_img.permute(2, 0, 1) #3,128,128

        # Prepare data and find model parameters
        det_img = torch.unsqueeze(det_img, 0).contiguous()

        io_binding = self.retinaface_model.io_binding()
        io_binding.bind_input(name='input.1', device_type='cuda', device_id=0, element_type=np.float32,  shape=det_img.size(), buffer_ptr=det_img.data_ptr())

        io_binding.bind_output('448', 'cuda')
        io_binding.bind_output('471', 'cuda')
        io_binding.bind_output('494', 'cuda')
        io_binding.bind_output('451', 'cuda')
        io_binding.bind_output('474', 'cuda')
        io_binding.bind_output('497', 'cuda')
        io_binding.bind_output('454', 'cuda')
        io_binding.bind_output('477', 'cuda')
        io_binding.bind_output('500', 'cuda')

        # Sync and run model
        self.syncvec.cpu()
        self.retinaface_model.run_with_iobinding(io_binding)

        net_outs = io_binding.copy_outputs_to_cpu()

        input_height = det_img.shape[2]
        input_width = det_img.shape[3]

        fmc = 3
        center_cache = {}
        scores_list = []
        bboxes_list = []
        kpss_list = []
        for idx, stride in enumerate([8, 16, 32]):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx+fmc]
            bbox_preds = bbox_preds * stride

            kps_preds = net_outs[idx+fmc*2] * stride
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in center_cache:
                anchor_centers = center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                anchor_centers = np.stack([anchor_centers]*2, axis=1).reshape( (-1,2) )
                if len(center_cache)<100:
                    center_cache[key] = anchor_centers

            pos_inds = np.where(scores>=score)[0]

            x1 = anchor_centers[:, 0] - bbox_preds[:, 0]
            y1 = anchor_centers[:, 1] - bbox_preds[:, 1]
            x2 = anchor_centers[:, 0] + bbox_preds[:, 2]
            y2 = anchor_centers[:, 1] + bbox_preds[:, 3]

            bboxes = np.stack([x1, y1, x2, y2], axis=-1)

            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            preds = []
            for i in range(0, kps_preds.shape[1], 2):
                px = anchor_centers[:, i%2] + kps_preds[:, i]
                py = anchor_centers[:, i%2+1] + kps_preds[:, i+1]

                preds.append(px)
                preds.append(py)
            kpss = np.stack(preds, axis=-1)
            #kpss = kps_preds
            kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        det_scale = det_scale.numpy()###

        bboxes = np.vstack(bboxes_list) / det_scale

        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        dets = pre_det
        thresh = 0.4
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scoresb = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        orderb = scoresb.argsort()[::-1]

        keep = []
        while orderb.size > 0:
            i = orderb[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[orderb[1:]])
            yy1 = np.maximum(y1[i], y1[orderb[1:]])
            xx2 = np.minimum(x2[i], x2[orderb[1:]])
            yy2 = np.minimum(y2[i], y2[orderb[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (areas[i] + areas[orderb[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            orderb = orderb[inds + 1]

        det = pre_det[keep, :]

        kpss = kpss[order,:,:]
        kpss = kpss[keep,:,:]

        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            det_img_center = det_img.shape[0] // 2, det_img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - det_img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - det_img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]

            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        score_values = det[:, 4]
        # delete score column
        det = np.delete(det, 4, 1)

        if use_landmark_detection and len(kpss) > 0:
            for i in range(kpss.shape[0]):
                landmark_kpss, landmark_scores = self.run_detect_landmark(img_landmark, det[i], kpss[i], landmark_detect_mode, landmark_score, from_points)
                if len(landmark_kpss) > 0:
                    if len(landmark_scores) > 0:
                        #print(np.mean(landmark_scores))
                        #print(np.mean(score_values[i]))
                        if np.mean(landmark_scores) > np.mean(score_values[i]):
                            kpss[i] = landmark_kpss
                    else:
                        kpss[i] = landmark_kpss

        return det, kpss

    def detect_retinaface2(self, img, max_num, score):

        # Resize image to fit within the input_size
        input_size = (640, 640)
        im_ratio = torch.div(img.size()[1], img.size()[2])

        # model_ratio = float(input_size[1]) / input_size[0]
        model_ratio = 1.0
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = torch.div(new_height, img.size()[1])

        resize = v2.Resize((new_height, new_width), antialias=True)
        img = resize(img)
        img = img.permute(1, 2, 0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.float32, device='cuda:0')
        det_img[:new_height, :new_width, :] = img

        # Switch to BGR and normalize
        det_img = det_img[:, :, [2, 1, 0]]
        det_img = torch.sub(det_img, 127.5)
        det_img = torch.div(det_img, 128.0)
        det_img = det_img.permute(2, 0, 1)  # 3,128,128

        # Prepare data and find model parameters
        det_img = torch.unsqueeze(det_img, 0).contiguous()

        io_binding = self.retinaface_model.io_binding()
        io_binding.bind_input(name='input.1', device_type='cuda', device_id=0, element_type=np.float32, shape=det_img.size(), buffer_ptr=det_img.data_ptr())

        io_binding.bind_output('448', 'cuda')
        io_binding.bind_output('471', 'cuda')
        io_binding.bind_output('494', 'cuda')
        io_binding.bind_output('451', 'cuda')
        io_binding.bind_output('474', 'cuda')
        io_binding.bind_output('497', 'cuda')
        io_binding.bind_output('454', 'cuda')
        io_binding.bind_output('477', 'cuda')
        io_binding.bind_output('500', 'cuda')

        # Sync and run model
        self.syncvec.cpu()
        self.retinaface_model.run_with_iobinding(io_binding)

        net_outs = io_binding.copy_outputs_to_cpu()

        input_height = det_img.shape[2]
        input_width = det_img.shape[3]

        fmc = 3
        center_cache = {}
        scores_list = []
        bboxes_list = []
        kpss_list = []
        for idx, stride in enumerate([8, 16, 32]):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + fmc]
            bbox_preds = bbox_preds * stride

            kps_preds = net_outs[idx + fmc * 2] * stride
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in center_cache:
                anchor_centers = center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                anchor_centers = np.stack([anchor_centers] * 2, axis=1).reshape((-1, 2))
                if len(center_cache) < 100:
                    center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= score)[0]

            x1 = anchor_centers[:, 0] - bbox_preds[:, 0]
            y1 = anchor_centers[:, 1] - bbox_preds[:, 1]
            x2 = anchor_centers[:, 0] + bbox_preds[:, 2]
            y2 = anchor_centers[:, 1] + bbox_preds[:, 3]

            bboxes = np.stack([x1, y1, x2, y2], axis=-1)

            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            preds = []
            for i in range(0, kps_preds.shape[1], 2):
                px = anchor_centers[:, i % 2] + kps_preds[:, i]
                py = anchor_centers[:, i % 2 + 1] + kps_preds[:, i + 1]

                preds.append(px)
                preds.append(py)
            kpss = np.stack(preds, axis=-1)
            # kpss = kps_preds
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)
        # result_boxes = cv2.dnn.NMSBoxes(bboxes_list, scores_list, 0.25, 0.45, 0.5)
        # print(result_boxes)
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        det_scale = det_scale.numpy()  ###

        bboxes = np.vstack(bboxes_list) / det_scale

        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]



        dets = pre_det
        thresh = 0.4
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scoresb = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        orderb = scoresb.argsort()[::-1]

        keep = []
        person_id = 0
        people = {}

        while orderb.size > 0:
            # Add first box in list
            i = orderb[0]
            keep.append(i)

            people[person_id] = orderb[0]


            # Find overlap of remaining boxes
            xx1 = np.maximum(x1[i], x1[orderb[1:]])
            yy1 = np.maximum(y1[i], y1[orderb[1:]])
            xx2 = np.minimum(x2[i], x2[orderb[1:]])
            yy2 = np.minimum(y2[i], y2[orderb[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h


            ovr = inter / (areas[i] + areas[orderb[1:]] - inter)


            inds0 = np.where(ovr > thresh)[0]
            people[person_id] = np.hstack((people[person_id], orderb[inds0+1])).astype(np.int, copy=False)


            # identify where there is no overlap (<thresh)
            inds = np.where(ovr <= thresh)[0]
            # print(len(inds))


            orderb = orderb[inds+1]
            person_id += 1



        det = pre_det[keep, :]


        kpss = kpss[order, :, :]
        # print('order', kpss)
        # kpss = kpss[keep, :, :]
        # print('keep',kpss)

        kpss_ave = []
        for person in people:
            # print(kpss[people[person], :, :])
            # print('mean', np.mean(kpss[people[person], :, :], axis=0))
            # print(kpss[people[person], :, :].shape)
            kpss_ave.append(np.mean(kpss[people[person], :, :], axis=0).tolist())


        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            det_img_center = det_img.shape[0] // 2, det_img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - det_img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - det_img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]

            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        # return kpss_ave

        # delete score column
        det = np.delete(det, 4, 1)

        return kpss_ave

    def detect_scrdf(self, img, max_num, score, use_landmark_detection, landmark_detect_mode, landmark_score, from_points):
        if use_landmark_detection:
            img_landmark = img.clone()

        # Resize image to fit within the input_size
        input_size = (640, 640)
        im_ratio = torch.div(img.size()[1], img.size()[2])

        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = torch.div(new_height,  img.size()[1])

        resize = v2.Resize((new_height, new_width), antialias=True)
        img = resize(img)
        img = img.permute(1,2,0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.float32, device='cuda:0')
        det_img[:new_height,:new_width,  :] = img

        # Switch to BGR and normalize
        det_img = det_img[:, :, [2,1,0]]
        det_img = torch.sub(det_img, 127.5)
        det_img = torch.div(det_img, 128.0)
        det_img = det_img.permute(2, 0, 1) #3,128,128

        # Prepare data and find model parameters
        det_img = torch.unsqueeze(det_img, 0).contiguous()
        input_name = self.scrdf_model.get_inputs()[0].name

        outputs = self.scrdf_model.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)

        io_binding = self.scrdf_model.io_binding()
        io_binding.bind_input(name=input_name, device_type='cuda', device_id=0, element_type=np.float32,  shape=det_img.size(), buffer_ptr=det_img.data_ptr())

        for i in range(len(output_names)):
            io_binding.bind_output(output_names[i], 'cuda')

        # Sync and run model
        syncvec = self.syncvec.cpu()
        self.scrdf_model.run_with_iobinding(io_binding)

        net_outs = io_binding.copy_outputs_to_cpu()

        input_height = det_img.shape[2]
        input_width = det_img.shape[3]

        fmc = 3
        center_cache = {}
        scores_list = []
        bboxes_list = []
        kpss_list = []
        for idx, stride in enumerate([8, 16, 32]):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx+fmc]
            bbox_preds = bbox_preds * stride

            kps_preds = net_outs[idx+fmc*2] * stride
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in center_cache:
                anchor_centers = center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                anchor_centers = np.stack([anchor_centers]*2, axis=1).reshape( (-1,2) )
                if len(center_cache)<100:
                    center_cache[key] = anchor_centers

            pos_inds = np.where(scores>=score)[0]

            x1 = anchor_centers[:, 0] - bbox_preds[:, 0]
            y1 = anchor_centers[:, 1] - bbox_preds[:, 1]
            x2 = anchor_centers[:, 0] + bbox_preds[:, 2]
            y2 = anchor_centers[:, 1] + bbox_preds[:, 3]

            bboxes = np.stack([x1, y1, x2, y2], axis=-1)

            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            preds = []
            for i in range(0, kps_preds.shape[1], 2):
                px = anchor_centers[:, i%2] + kps_preds[:, i]
                py = anchor_centers[:, i%2+1] + kps_preds[:, i+1]

                preds.append(px)
                preds.append(py)
            kpss = np.stack(preds, axis=-1)
            #kpss = kps_preds
            kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        det_scale = det_scale.numpy()###

        bboxes = np.vstack(bboxes_list) / det_scale

        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        dets = pre_det
        thresh = 0.4
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scoresb = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        orderb = scoresb.argsort()[::-1]

        keep = []
        while orderb.size > 0:
            i = orderb[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[orderb[1:]])
            yy1 = np.maximum(y1[i], y1[orderb[1:]])
            xx2 = np.minimum(x2[i], x2[orderb[1:]])
            yy2 = np.minimum(y2[i], y2[orderb[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (areas[i] + areas[orderb[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            orderb = orderb[inds + 1]

        det = pre_det[keep, :]

        kpss = kpss[order,:,:]
        kpss = kpss[keep,:,:]

        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                    det[:, 1])
            det_img_center = det_img.shape[0] // 2, det_img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - det_img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - det_img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]

            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        score_values = det[:, 4]
        # delete score column
        det = np.delete(det, 4, 1)

        if use_landmark_detection and len(kpss) > 0:
            for i in range(kpss.shape[0]):
                landmark_kpss, landmark_scores = self.run_detect_landmark(img_landmark, det[i], kpss[i], landmark_detect_mode, landmark_score, from_points)
                if len(landmark_kpss) > 0:
                    if len(landmark_scores) > 0:
                        #print(np.mean(landmark_scores))
                        #print(np.mean(score_values[i]))
                        if np.mean(landmark_scores) > np.mean(score_values[i]):
                            kpss[i] = landmark_kpss
                    else:
                        kpss[i] = landmark_kpss

        return det, kpss

    def detect_yoloface(self, img, max_num, score, use_landmark_detection, landmark_detect_mode, landmark_score, from_points):
        if use_landmark_detection:
            img_landmark = img.clone()

        height = img.size(dim=1)
        width = img.size(dim=2)
        length = max((height, width))

        image = torch.zeros((length, length, 3), dtype=torch.uint8, device='cuda')
        img = img.permute(1,2,0)

        image[0:height, 0:width] = img
        scale = length/640.0
        image = torch.div(image, 255.0)

        t640 = v2.Resize((640, 640), antialias=False)
        image = image.permute(2, 0, 1)
        image = t640(image)

        image = torch.unsqueeze(image, 0).contiguous()

        io_binding = self.yoloface_model.io_binding()
        io_binding.bind_input(name='images', device_type='cuda', device_id=0, element_type=np.float32,  shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output('output0', 'cuda')

        # Sync and run model
        self.syncvec.cpu()
        self.yoloface_model.run_with_iobinding(io_binding)

        net_outs = io_binding.copy_outputs_to_cpu()

        outputs = np.squeeze(net_outs).T

        bbox_raw, score_raw, kps_raw = np.split(outputs, [4, 5], axis=1)

        bbox_list = []
        score_list = []
        kps_list = []
        keep_indices = np.where(score_raw > score)[0]

        if keep_indices.any():
            bbox_raw, kps_raw, score_raw = bbox_raw[keep_indices], kps_raw[keep_indices], score_raw[keep_indices]

            bbox_raw = bbox_raw * scale

            for bbox in bbox_raw:
                bbox_list.append(np.array([(bbox[0]-bbox[2]/2), (bbox[1]-bbox[3]/2), (bbox[0]+bbox[2]/2), (bbox[1]+bbox[3]/2)]))

            kps_raw = kps_raw * scale

            for kps in kps_raw:
                indexes = np.arange(0, len(kps), 3)
                temp_kps = []
                for index in indexes:
                    temp_kps.append([kps[index], kps[index + 1]])
                kps_list.append(np.array(temp_kps))
            score_list = score_raw.ravel().tolist()

        result_boxes = cv2.dnn.NMSBoxes(bbox_list, score_list, 0.25, 0.45, 0.5)

        bboxes_list = []
        kpss_list = []
        for r in result_boxes:
            if r==max_num:
                break
            if use_landmark_detection and len(kps_list[r]) > 0:
                landmark_kpss, landmark_scores = self.run_detect_landmark(img_landmark, bbox_list[r], kps_list[r], landmark_detect_mode, landmark_score, from_points)
                if len(landmark_kpss) > 0:
                    if len(landmark_scores) > 0:
                        #print(np.mean(landmark_scores))
                        #print(np.mean(score_list[r]))
                        if np.mean(landmark_scores) > np.mean(score_list[r]):
                            kps_list[r] = landmark_kpss
                    else:
                        kps_list[r] = landmark_kpss

            bboxes_list.append(bbox_list[r])
            kpss_list.append(kps_list[r])

        return np.array(bboxes_list), np.array(kpss_list)

    def detect_yoloface2(self, image_in, max_num, score):
        img = image_in.detach().clone()

        height = img.size(dim=1)
        width = img.size(dim=2)
        length = max((height, width))

        image = torch.zeros((length, length, 3), dtype=torch.uint8,
                            device='cuda')
        img = img.permute(1, 2, 0)

        image[0:height, 0:width] = img
        scale = length / 640.0
        image = torch.div(image, 255.0)

        t640 = v2.Resize((640, 640), antialias=False)
        image = image.permute(2, 0, 1)
        image = t640(image)

        image = torch.unsqueeze(image, 0).contiguous()

        io_binding = self.yoloface_model.io_binding()
        io_binding.bind_input(name='images', device_type='cuda', device_id=0,
                              element_type=np.float32, shape=image.size(),
                              buffer_ptr=image.data_ptr())
        io_binding.bind_output('output0', 'cuda')

        # Sync and run model
        self.syncvec.cpu()
        self.yoloface_model.run_with_iobinding(io_binding)

        net_outs = io_binding.copy_outputs_to_cpu()

        outputs = np.squeeze(net_outs).T

        bbox_raw, score_raw, kps_raw = np.split(outputs, [4, 5], axis=1)

        bbox_list = []
        score_list = []
        kps_list = []
        keep_indices = np.where(score_raw > score)[0]

        if keep_indices.any():
            bbox_raw, kps_raw, score_raw = bbox_raw[keep_indices], kps_raw[
                keep_indices], score_raw[keep_indices]
            for bbox in bbox_raw:
                bbox_list.append(np.array(
                    [(bbox[0] - bbox[2] / 2), (bbox[1] - bbox[3] / 2),
                     (bbox[0] + bbox[2] / 2), (bbox[1] + bbox[3] / 2)]))
            kps_raw = kps_raw * scale

            for kps in kps_raw:
                indexes = np.arange(0, len(kps), 3)
                temp_kps = []
                for index in indexes:
                    temp_kps.append([kps[index], kps[index + 1]])
                kps_list.append(np.array(temp_kps))
            score_list = score_raw.ravel().tolist()

        result_boxes = cv2.dnn.NMSBoxes(bbox_list, score_list, 0.25, 0.45, 0.5)

        result = []
        for r in result_boxes:
            if r == max_num:
                break
            bbox_list = bbox_list[r]
            result.append(kps_list[r])
        bbox_list = bbox_list*scale
        # print(bbox_list)
        # print(bbox_list*scale)

        # img = image_in.detach().clone()
        # test = image_in.permute(1, 2, 0)
        # test = test.cpu().numpy()
        # cv2.imwrite('1.jpg', test)

        # b_scale = 50
        # bbox_list[0] = bbox_list[0] - b_scale
        # bbox_list[1] = bbox_list[1] - b_scale
        # bbox_list[2] = bbox_list[2] + b_scale
        # bbox_list[3] = bbox_list[3] + b_scale

        img = image_in.detach().clone()

        img = img[:, int(bbox_list[1]):int(bbox_list[3]), int(bbox_list[0]):int(bbox_list[2])]
        # print(img.size())


        height = img.size(dim=1)
        width = img.size(dim=2)
        length = max((height, width))

        image = torch.zeros((length, length, 3), dtype=torch.uint8, device='cuda')
        img = img.permute(1,2,0)

        image[0:height, 0:width] = img
        scale = length/192
        image = torch.div(image, 255.0)


        t192 = v2.Resize((192, 192), antialias=False)
        image = image.permute(2, 0, 1)
        image = t192(image)

        test = image_in.detach().clone().permute(1, 2, 0)
        test = test.cpu().numpy()

        input_mean = 0.0
        input_std = 1.0

        self.lmk_dim = 2
        self.lmk_num = 106

        bbox = bbox_list
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = 192 / (max(w, h) * 1.5)
        # print('param:', img.shape, bbox, center, self.input_size, _scale, rotate)
        aimg, M = self.transform(test, center, 192, _scale, rotate)
        input_size = tuple(aimg.shape[0:2][::-1])
        # assert input_size==self.input_size
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / input_std, input_size, ( input_mean, input_mean, input_mean), swapRB=True)
        pred = self.insight106_model.run(['fc1'], {'data': blob})[0][0]
        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))
        if self.lmk_num < pred.shape[0]:
            pred = pred[self.lmk_num * -1:, :]
        pred[:, 0:2] += 1
        pred[:, 0:2] *= 96
        if pred.shape[1] == 3:
            pred[:, 2] *= (106)

        IM = cv2.invertAffineTransform(M)
        pred = self.trans_points2d(pred, IM)
        # face[self.taskname] = pred
        # if self.require_pose:
        #     P = transform.estimate_affine_matrix_3d23d(self.mean_lmk, pred)
        #     s, R, t = transform.P2sRt(P)
        #     rx, ry, rz = transform.matrix2angle(R)
        #     pose = np.array([rx, ry, rz], dtype=np.float32)
        #     face['pose'] = pose  # pitch, yaw, roll
        # print(pred.shape)
        # print(pred)

        for point in pred:
            test[int(point[1])] [int(point[0])] [0] = 255
            test[int(point[1])] [int(point[0])] [1] = 255
            test[int(point[1])] [int(point[0])] [2] = 255
        cv2.imwrite('2.jpg', test)

        predd = []
        predd.append(pred[38])
        predd.append(pred[88])
        # predd.append(pred[86])
        # predd.append(pred[52])
        # predd.append(pred[61])

        predd.append(kps_list[0][2])
        predd.append(kps_list[0][3])
        predd.append(kps_list[0][4])

        # for point in predd:
        #     test[int(point[1])] [int(point[0])] [0] = 255
        #     test[int(point[1])] [int(point[0])] [1] = 255
        #     test[int(point[1])] [int(point[0])] [2] = 255
        # cv2.imwrite('2.jpg', test)
        preddd=[]
        preddd.append(predd)
        return np.array(preddd)
    def transform(self, data, center, output_size, scale, rotation):
        scale_ratio = scale
        rot = float(rotation) * np.pi / 180.0
        # translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
        t1 = trans.SimilarityTransform(scale=scale_ratio)
        cx = center[0] * scale_ratio
        cy = center[1] * scale_ratio
        t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
        t3 = trans.SimilarityTransform(rotation=rot)
        t4 = trans.SimilarityTransform(translation=(output_size / 2,
                                                    output_size / 2))
        t = t1 + t2 + t3 + t4
        M = t.params[0:2]
        cropped = cv2.warpAffine(data,
                                 M, (output_size, output_size),
                                 borderValue=0.0)
        return cropped, M

    def trans_points2d(self, pts, M):
        new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
        for i in range(pts.shape[0]):
            pt = pts[i]
            new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
            new_pt = np.dot(M, new_pt)
            # print('new_pt', new_pt.shape, new_pt)
            new_pts[i] = new_pt[0:2]

        return new_pts

        # image = torch.unsqueeze(image, 0).contiguous()
        #
        # io_binding = self.insight106_model.io_binding()
        # io_binding.bind_input(name='data', device_type='cuda', device_id=0, element_type=np.float32,  shape=image.size(), buffer_ptr=image.data_ptr())
        # io_binding.bind_output('fc1', 'cuda')
        #
        # # Sync and run model
        # self.syncvec.cpu()
        # self.insight106_model.run_with_iobinding(io_binding)
        #
        # net_outs = io_binding.copy_outputs_to_cpu()
        # print(net_outs)
        # net_outs[0][0] = net_outs[0][0]+1.
        # net_outs[0][0] = net_outs[0][0]/2.
        # net_outs[0][0] = net_outs[0][0]*96
        #
        # # net_outs[0] = net_outs[0]*scale
        # # print(net_outs)
        # test=test*255.0
        # for i in range(0, len(net_outs[0][0]), 2):
        #     test[int(net_outs[0][0][i+1])] [int(net_outs[0][0][i])] [0] = 255
        #     test[int(net_outs[0][0][i+1])] [int(net_outs[0][0][i])] [1] = 255
        #     test[int(net_outs[0][0][i+1])] [int(net_outs[0][0][i])] [2] = 255
        # cv2.imwrite('2.jpg', test)
        #
        # return np.array(result)
    def detect_yunet(self, img, max_num, score, use_landmark_detection, landmark_detect_mode, landmark_score, from_points):
        if use_landmark_detection:
            img_landmark = img.clone()

        height = img.size(dim=1)
        width = img.size(dim=2)
        input_size = (640, 640)
        im_ratio = float(height) / width
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / height

        t640 = v2.Resize((new_height, new_width), antialias=False)
        img = t640(img)

        # Switch to BGR
        img = img.permute(1,2,0)
        img = img[:, :, [2,1,0]]

        image = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.uint8, device='cuda')
        image[:new_height, :new_width, :] = img

        image = image.permute(2, 0, 1)
        image = torch.unsqueeze(image, 0).contiguous()
        image = image.to(dtype=torch.float32)

        input_name = self.yunet_model.get_inputs()[0].name
        outputs = self.yunet_model.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)

        io_binding = self.yunet_model.io_binding() 
        io_binding.bind_input(name=input_name, device_type='cuda', device_id=0, element_type=np.float32,  shape=image.size(), buffer_ptr=image.data_ptr())

        for i in range(len(output_names)):
            io_binding.bind_output(output_names[i], 'cuda')

        # Sync and run model
        syncvec = self.syncvec.cpu()
        self.yunet_model.run_with_iobinding(io_binding)
        net_outs = io_binding.copy_outputs_to_cpu()

        strides = [8, 16, 32]
        scores, bboxes, kpss = [], [], []
        for idx, stride in enumerate(strides):
            cls_pred = net_outs[idx].reshape(-1, 1)
            obj_pred = net_outs[idx + len(strides)].reshape(-1, 1)
            reg_pred = net_outs[idx + len(strides) * 2].reshape(-1, 4)
            kps_pred = net_outs[idx + len(strides) * 3].reshape(
                -1, 5 * 2)

            anchor_centers = np.stack(
                np.mgrid[:(input_size[1] // stride), :(input_size[0] //
                                                       stride)][::-1],
                axis=-1)
            anchor_centers = (anchor_centers * stride).astype(
                np.float32).reshape(-1, 2)

            bbox_cxy = reg_pred[:, :2] * stride + anchor_centers[:]
            bbox_wh = np.exp(reg_pred[:, 2:]) * stride
            tl_x = (bbox_cxy[:, 0] - bbox_wh[:, 0] / 2.)
            tl_y = (bbox_cxy[:, 1] - bbox_wh[:, 1] / 2.)
            br_x = (bbox_cxy[:, 0] + bbox_wh[:, 0] / 2.)
            br_y = (bbox_cxy[:, 1] + bbox_wh[:, 1] / 2.)

            bboxes.append(np.stack([tl_x, tl_y, br_x, br_y], -1))
            # for nk in range(5):
            per_kps = np.concatenate(
                [((kps_pred[:, [2 * i, 2 * i + 1]] * stride) + anchor_centers)
                 for i in range(5)],
                axis=-1)

            kpss.append(per_kps)
            scores.append(cls_pred * obj_pred)

        scores = np.concatenate(scores, axis=0).reshape(-1)
        bboxes = np.concatenate(bboxes, axis=0)
        kpss = np.concatenate(kpss, axis=0)
        score_mask = (scores > score)
        scores = scores[score_mask]
        bboxes = bboxes[score_mask]
        kpss = kpss[score_mask]

        bboxes /= det_scale
        kpss /= det_scale
        pre_det = np.hstack((bboxes, scores[:, None]))

        dets = pre_det
        thresh = 0.4
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scoresb = dets[:, -1]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scoresb.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        kpss = kpss[keep, :]
        bboxes = pre_det[keep, :]
        score_values = bboxes[:, 4]

        bbox_list = []
        kps_list = []
        for i in range(bboxes.shape[0]):
            if i==max_num:
                    break
            box = np.array((bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]))
            bbox_list.append(box)

            if kpss is not None:
                kps = kpss[i].reshape(-1, 2)
                if use_landmark_detection and len(kps) > 0:
                    landmark_kpss, landmark_scores = self.run_detect_landmark(img_landmark, box, kps, landmark_detect_mode, landmark_score, from_points)
                    if len(landmark_kpss) > 0:
                        if len(landmark_scores) > 0:
                            #print(np.mean(landmark_scores))
                            #print(np.mean(score_values[i]))
                            if np.mean(landmark_scores) > np.mean(score_values[i]):
                                kps = landmark_kpss
                        else:
                            kps = landmark_kpss

                kps_list.append(kps)

        return np.array(bbox_list), np.array(kps_list)

    def detect_face_landmark_5(self, img, bbox, det_kpss, from_points=False):
        if from_points == False:
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = 512.0  / (max(w, h)*1.5)
            image, M = faceutil.transform(img, center, 512, _scale, rotate)
        else:
            image, M = faceutil.warp_face_by_face_landmark_5(img, det_kpss, 512, mode='arcface128', interpolation=v2.InterpolationMode.BILINEAR)

        image = image.permute(1,2,0)

        mean = torch.tensor([104, 117, 123], dtype=torch.float32, device='cuda')
        image = torch.sub(image, mean)

        image = image.permute(2,0,1)
        image = torch.reshape(image, (1, 3, 512, 512))

        height, width = (512, 512)
        tmp = [width, height, width, height, width, height, width, height, width, height]
        scale1 = torch.tensor(tmp, dtype=torch.float32, device='cuda')

        conf = torch.empty((1,10752,2), dtype=torch.float32, device='cuda').contiguous()
        landmarks = torch.empty((1,10752,10), dtype=torch.float32, device='cuda').contiguous()

        io_binding = self.resnet50_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='conf', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,10752,2), buffer_ptr=conf.data_ptr())
        io_binding.bind_output(name='landmarks', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,10752,10), buffer_ptr=landmarks.data_ptr())

        torch.cuda.synchronize('cuda')
        self.resnet50_model.run_with_iobinding(io_binding)

        scores = torch.squeeze(conf)[:, 1]
        priors = torch.tensor(self.anchors).view(-1, 4)
        priors = priors.to('cuda')

        pre = torch.squeeze(landmarks, 0)

        tmp = (priors[:, :2] + pre[:, :2] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 2:4] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 4:6] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 6:8] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 8:10] * 0.1 * priors[:, 2:])
        landmarks = torch.cat(tmp, dim=1)
        landmarks = torch.mul(landmarks, scale1)

        landmarks = landmarks.cpu().numpy()

        # ignore low scores
        score=.1
        inds = torch.where(scores>score)[0]
        inds = inds.cpu().numpy()  
        scores = scores.cpu().numpy()  

        landmarks, scores = landmarks[inds], scores[inds]    

        # sort
        order = scores.argsort()[::-1]

        if len(order) > 0:
            landmarks = landmarks[order][0]
            scores = scores[order][0]

            landmarks = np.array([[landmarks[i], landmarks[i + 1]] for i in range(0,10,2)])

            IM = faceutil.invertAffineTransform(M)
            landmarks = faceutil.trans_points2d(landmarks, IM)
            scores = np.array([scores])
            
            #faceutil.test_bbox_landmarks(img, bbox, landmarks)
            #print(scores)
                    
            return landmarks, scores

        return [], []

    def detect_face_landmark_68(self, img, bbox, det_kpss, convert68_5=True, from_points=False):
        if from_points == False:
            crop_image, affine_matrix = faceutil.warp_face_by_bounding_box_for_landmark_68(img, bbox, (256, 256))
        else:
            crop_image, affine_matrix = faceutil.warp_face_by_face_landmark_5(img, det_kpss, 256, mode='arcface128', interpolation=v2.InterpolationMode.BILINEAR)
        '''
        cv2.imshow('image', crop_image.permute(1, 2, 0).to('cpu').numpy())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        crop_image = crop_image.to(dtype=torch.float32)
        crop_image = torch.div(crop_image, 255.0)
        crop_image = torch.unsqueeze(crop_image, 0).contiguous()

        io_binding = self.face_landmark_68_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32,  shape=crop_image.size(), buffer_ptr=crop_image.data_ptr())

        io_binding.bind_output('landmarks_xyscore', 'cuda')
        io_binding.bind_output('heatmaps', 'cuda')

        # Sync and run model
        syncvec = self.syncvec.cpu()
        self.face_landmark_68_model.run_with_iobinding(io_binding)
        net_outs = io_binding.copy_outputs_to_cpu()
        face_landmark_68 = net_outs[0]
        face_heatmap = net_outs[1]

        face_landmark_68 = face_landmark_68[:, :, :2][0] / 64.0
        face_landmark_68 = face_landmark_68.reshape(1, -1, 2) * 256.0
        face_landmark_68 = cv2.transform(face_landmark_68, cv2.invertAffineTransform(affine_matrix))

        face_landmark_68 = face_landmark_68.reshape(-1, 2)
        face_landmark_68_score = np.amax(face_heatmap, axis = (2, 3))
        face_landmark_68_score = face_landmark_68_score.reshape(-1, 1)

        if convert68_5:
            face_landmark_68, face_landmark_68_score = faceutil.convert_face_landmark_68_to_5(face_landmark_68, face_landmark_68_score)

        #faceutil.test_bbox_landmarks(img, bbox, face_landmark_68)

        return face_landmark_68, face_landmark_68_score

    def detect_face_landmark_3d68(self, img, bbox, det_kpss, convert68_5=True, from_points=False):
        if from_points == False:
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = 192  / (max(w, h)*1.5)
            #print('param:', img.size(), bbox, center, (192, 192), _scale, rotate)
            aimg, M = faceutil.transform(img, center, 192, _scale, rotate)
        else:
            aimg, M = faceutil.warp_face_by_face_landmark_5(img, det_kpss, image_size=192, mode='arcface128', interpolation=v2.InterpolationMode.BILINEAR)
        '''
        cv2.imshow('image', aimg.permute(1.2.0).to('cpu').numpy())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        aimg = torch.unsqueeze(aimg, 0).contiguous()
        aimg = aimg.to(dtype=torch.float32)
        aimg = self.normalize(aimg)
        io_binding = self.face_landmark_3d68_model.io_binding()
        io_binding.bind_input(name='data', device_type='cuda', device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

        io_binding.bind_output('fc1', 'cuda')

        # Sync and run model
        syncvec = self.syncvec.cpu()
        self.face_landmark_3d68_model.run_with_iobinding(io_binding)
        pred = io_binding.copy_outputs_to_cpu()[0][0]

        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))
        if 68 < pred.shape[0]:
            pred = pred[68*-1:,:]
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (192 // 2)
        if pred.shape[1] == 3:
            pred[:, 2] *= (192 // 2)

        #IM = cv2.invertAffineTransform(M)
        IM = faceutil.invertAffineTransform(M)
        pred = faceutil.trans_points3d(pred, IM)

        # at moment we don't use 3d points
        '''
        P = faceutil.estimate_affine_matrix_3d23d(self.mean_lmk, pred)
        s, R, t = faceutil.P2sRt(P)
        rx, ry, rz = faceutil.matrix2angle(R)
        pose = np.array( [rx, ry, rz], dtype=np.float32 ) #pitch, yaw, roll
        '''

        # convert from 3d68 to 2d68 keypoints
        landmark2d68 = np.array(pred[:, [0, 1]])

        if convert68_5:
            # convert from 68 to 5 keypoints
            landmark2d68, _ = faceutil.convert_face_landmark_68_to_5(landmark2d68, [])

        #faceutil.test_bbox_landmarks(img, bbox, landmark2d68)

        return landmark2d68, []

    def detect_face_landmark_98(self, img, bbox, det_kpss, convert98_5=True, from_points=False):
        if from_points == False:
            crop_image, detail = faceutil.warp_face_by_bounding_box_for_landmark_98(img, bbox, (256, 256))
        else:
            crop_image, M = faceutil.warp_face_by_face_landmark_5(img, det_kpss, image_size=256, mode='arcface128', interpolation=v2.InterpolationMode.BILINEAR)
            #crop_image2 = crop_image.clone()
            h, w = (crop_image.size(dim=1), crop_image.size(dim=2))
        '''
        cv2.imshow('image', crop_image.permute(1, 2, 0).to('cpu').numpy())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        landmark = []
        landmark_score = []
        if crop_image is not None:
            crop_image = crop_image.to(dtype=torch.float32)
            crop_image = torch.div(crop_image, 255.0)
            crop_image = torch.unsqueeze(crop_image, 0).contiguous()

            io_binding = self.face_landmark_98_model.io_binding()
            io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32,  shape=crop_image.size(), buffer_ptr=crop_image.data_ptr())

            io_binding.bind_output('landmarks_xyscore', 'cuda')

            # Sync and run model
            syncvec = self.syncvec.cpu()
            self.face_landmark_98_model.run_with_iobinding(io_binding)
            landmarks_xyscore = io_binding.copy_outputs_to_cpu()[0]

            if len(landmarks_xyscore) > 0:
                for one_face_landmarks in landmarks_xyscore:
                    landmark_score = one_face_landmarks[:, [2]].reshape(-1)
                    landmark = one_face_landmarks[:, [0, 1]].reshape(-1,2)

                    ##recorver, and grouped as [98,2]
                    if from_points == False:
                        landmark[:, 0] = landmark[:, 0] * detail[1] + detail[3] - detail[4]
                        landmark[:, 1] = landmark[:, 1] * detail[0] + detail[2] - detail[4]
                    else:
                        landmark[:, 0] = landmark[:, 0] * w
                        landmark[:, 1] = landmark[:, 1] * h
                        #lmk = landmark.copy()
                        #lmk_score = landmark_score.copy()

                        #IM = cv2.invertAffineTransform(M)
                        IM = faceutil.invertAffineTransform(M)
                        landmark = faceutil.trans_points2d(landmark, IM)

                    if convert98_5:
                        landmark, landmark_score = faceutil.convert_face_landmark_98_to_5(landmark, landmark_score)
                        #lmk, lmk_score = faceutil.convert_face_landmark_98_to_5(lmk, lmk_score)

                    #faceutil.test_bbox_landmarks(crop_image2, [], lmk)
                    #faceutil.test_bbox_landmarks(img, bbox, landmark)
                    #faceutil.test_bbox_landmarks(img, bbox, det_kpss)

        return landmark, landmark_score

    def detect_face_landmark_106(self, img, bbox, det_kpss, convert106_5=True, from_points=False):
        if from_points == False:
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = 192  / (max(w, h)*1.5)
            #print('param:', img.size(), bbox, center, (192, 192), _scale, rotate)
            aimg, M = faceutil.transform(img, center, 192, _scale, rotate)
        else:
            aimg, M = faceutil.warp_face_by_face_landmark_5(img, det_kpss, image_size=192, mode='arcface128', interpolation=v2.InterpolationMode.BILINEAR)
        '''
        cv2.imshow('image', aimg.permute(1.2.0).to('cpu').numpy())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        aimg = torch.unsqueeze(aimg, 0).contiguous()
        aimg = aimg.to(dtype=torch.float32)
        aimg = self.normalize(aimg)
        io_binding = self.face_landmark_106_model.io_binding()
        io_binding.bind_input(name='data', device_type='cuda', device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

        io_binding.bind_output('fc1', 'cuda')

        # Sync and run model
        syncvec = self.syncvec.cpu()
        self.face_landmark_106_model.run_with_iobinding(io_binding)
        pred = io_binding.copy_outputs_to_cpu()[0][0]

        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))

        if 106 < pred.shape[0]:
            pred = pred[106*-1:,:]

        pred[:, 0:2] += 1
        pred[:, 0:2] *= (192 // 2)
        if pred.shape[1] == 3:
            pred[:, 2] *= (192 // 2)

        #IM = cv2.invertAffineTransform(M)
        IM = faceutil.invertAffineTransform(M)
        pred = faceutil.trans_points(pred, IM)

        if pred is not None:
            if convert106_5:
                # convert from 106 to 5 keypoints
                pred = faceutil.convert_face_landmark_106_to_5(pred)

        #faceutil.test_bbox_landmarks(img, bbox, pred)

        return pred, []

    def detect_face_landmark_478(self, img, bbox, det_kpss, convert478_5=True, from_points=False):
        if from_points == False:
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = 256.0  / (max(w, h)*1.5)
            #print('param:', img.size(), bbox, center, (192, 192), _scale, rotate)
            aimg, M = faceutil.transform(img, center, 256, _scale, rotate)
        else:
            aimg, M = faceutil.warp_face_by_face_landmark_5(img, det_kpss, 256, mode='arcfacemap', interpolation=v2.InterpolationMode.BILINEAR)
            #aimg2 = aimg.clone()
        '''
        cv2.imshow('image', aimg.permute(1,2,0).to('cpu').numpy())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        aimg = torch.unsqueeze(aimg, 0).contiguous()
        aimg = aimg.to(dtype=torch.float32)
        aimg = torch.div(aimg, 255.0)
        io_binding = self.face_landmark_478_model.io_binding()
        io_binding.bind_input(name='input_12', device_type='cuda', device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

        io_binding.bind_output('Identity', 'cuda')
        io_binding.bind_output('Identity_1', 'cuda')
        io_binding.bind_output('Identity_2', 'cuda')

        # Sync and run model
        syncvec = self.syncvec.cpu()
        self.face_landmark_478_model.run_with_iobinding(io_binding)
        landmarks, faceflag, blendshapes = io_binding.copy_outputs_to_cpu()
        landmarks = landmarks.reshape( (1,478,3))

        landmark = []
        landmark_score = []
        if len(landmarks) > 0:
            for one_face_landmarks in landmarks:
                #lmk = one_face_landmarks.copy()
                landmark = one_face_landmarks
                #IM = cv2.invertAffineTransform(M)
                IM = faceutil.invertAffineTransform(M)
                landmark = faceutil.trans_points3d(landmark, IM)
                '''
                P = faceutil.estimate_affine_matrix_3d23d(self.mean_lmk, landmark)
                s, R, t = faceutil.P2sRt(P)
                rx, ry, rz = faceutil.matrix2angle(R)
                pose = np.array( [rx, ry, rz], dtype=np.float32 ) #pitch, yaw, roll
                '''
                landmark = landmark[:, [0, 1]].reshape(-1,2)
                #lmk = lmk[:, [0, 1]].reshape(-1,2)

                #get scores
                landmark_for_score = landmark[self.LandmarksSubsetIdxs]
                landmark_for_score = landmark_for_score[:, :2]
                landmark_for_score = np.expand_dims(landmark_for_score, axis=0)
                landmark_for_score = landmark_for_score.astype(np.float32)
                landmark_for_score = torch.from_numpy(landmark_for_score).to('cuda')
    
                io_binding_bs = self.face_blendshapes_model.io_binding()
                io_binding_bs.bind_input(name='input_points', device_type='cuda', device_id=0, element_type=np.float32,  shape=tuple(landmark_for_score.shape), buffer_ptr=landmark_for_score.data_ptr())
                io_binding_bs.bind_output('output', 'cuda')

                # Sync and run model
                syncvec = self.syncvec.cpu()
                self.face_blendshapes_model.run_with_iobinding(io_binding_bs)
                landmark_score = io_binding_bs.copy_outputs_to_cpu()[0]
                
                if convert478_5:
                    # convert from 478 to 5 keypoints
                    landmark = faceutil.convert_face_landmark_478_to_5(landmark)
                    #lmk = faceutil.convert_face_landmark_478_to_5(lmk)

                #faceutil.test_bbox_landmarks(aimg2, [], lmk)
                #faceutil.test_bbox_landmarks(img, bbox, landmark)
                #faceutil.test_bbox_landmarks(img, bbox, det_kpss)

        #return landmark, landmark_score
        return landmark, []

    def recognize(self, recognition_model, img, face_kps, similarity_type):
        if similarity_type == 'Optimal':
            # Find transform & Transform
            img, _ = faceutil.warp_face_by_face_landmark_5(img, face_kps, mode='arcfacemap', interpolation=v2.InterpolationMode.BILINEAR)
        elif similarity_type == 'Pearl':
            # Find transform
            dst = self.arcface_dst.copy()
            dst[:, 0] += 8.0

            tform = trans.SimilarityTransform()
            tform.estimate(face_kps, dst)

            # Transform
            img = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) ) 
            img = v2.functional.crop(img, 0,0, 128, 128)
            img = v2.Resize((112, 112), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)(img)
        else:
            # Find transform 
            tform = trans.SimilarityTransform()
            tform.estimate(face_kps, self.arcface_dst)

            # Transform
            img = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) ) 
            img = v2.functional.crop(img, 0,0, 112, 112)

        if recognition_model == self.recognition_model or recognition_model == self.recognition_simswap_model:
            # Switch to BGR and normalize
            img = img.permute(1,2,0) #112,112,3
            cropped_image = img
            img = img[:, :, [2,1,0]]
            img = torch.sub(img, 127.5)
            img = torch.div(img, 127.5)
            img = img.permute(2, 0, 1) #3,112,112
        else:
            cropped_image = img.permute(1,2,0) #112,112,3
            # Converti a float32 e normalizza
            img = torch.div(img.float(), 127.5)
            img = torch.sub(img, 1)
            # img è già in RGB quindi nessuna conversione.
            #img = img[[2, 1, 0], :, :] # Inverte i canali da BGR a RGB (assumendo che l'input sia BGR)

        # Prepare data and find model parameters
        img = torch.unsqueeze(img, 0).contiguous()
        input_name = recognition_model.get_inputs()[0].name

        outputs = recognition_model.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)

        io_binding = recognition_model.io_binding()
        io_binding.bind_input(name=input_name, device_type='cuda', device_id=0, element_type=np.float32,  shape=img.size(), buffer_ptr=img.data_ptr())

        for i in range(len(output_names)):
            io_binding.bind_output(output_names[i], 'cuda')

        # Sync and run model
        self.syncvec.cpu()
        recognition_model.run_with_iobinding(io_binding)

        # Return embedding
        return np.array(io_binding.copy_outputs_to_cpu()).flatten(), cropped_image

    def resnet50(self, image, score=.5):
        if not self.resnet50_model:
            self.resnet50_model = onnxruntime.InferenceSession("./models/res50.onnx", providers=self.providers)

            feature_maps = [[64, 64], [32, 32], [16, 16]]
            min_sizes = [[16, 32], [64, 128], [256, 512]]
            steps = [8, 16, 32]
            image_size = 512

            for k, f in enumerate(feature_maps):
                min_size_array = min_sizes[k]
                for i, j in product(range(f[0]), range(f[1])):
                    for min_size in min_size_array:
                        s_kx = min_size / image_size
                        s_ky = min_size / image_size
                        dense_cx = [x * steps[k] / image_size for x in [j + 0.5]]
                        dense_cy = [y * steps[k] / image_size for y in [i + 0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            self.anchors += [cx, cy, s_kx, s_ky]

        # image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        image = image.permute(1,2,0)

        # image = image - [104, 117, 123]
        mean = torch.tensor([104, 117, 123], dtype=torch.float32, device='cuda')
        image = torch.sub(image, mean)

        # image = image.transpose(2, 0, 1)
        # image = np.float32(image[np.newaxis,:,:,:])
        image = image.permute(2,0,1)
        image = torch.reshape(image, (1, 3, 512, 512))

        height, width = (512, 512)
        tmp = [width, height, width, height, width, height, width, height, width, height]
        scale1 = torch.tensor(tmp, dtype=torch.float32, device='cuda')

        # ort_inputs = {"input": image}
        conf = torch.empty((1,10752,2), dtype=torch.float32, device='cuda').contiguous()
        landmarks = torch.empty((1,10752,10), dtype=torch.float32, device='cuda').contiguous()

        io_binding = self.resnet50_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='conf', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,10752,2), buffer_ptr=conf.data_ptr())
        io_binding.bind_output(name='landmarks', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,10752,10), buffer_ptr=landmarks.data_ptr())

        # _, conf, landmarks = self.resnet_model.run(None, ort_inputs)
        torch.cuda.synchronize('cuda')
        self.resnet50_model.run_with_iobinding(io_binding)

        # conf = torch.from_numpy(conf)
        # scores = conf.squeeze(0).numpy()[:, 1]
        scores = torch.squeeze(conf)[:, 1]

        # landmarks = torch.from_numpy(landmarks)
        # landmarks = landmarks.to('cuda')

        priors = torch.tensor(self.anchors).view(-1, 4)
        priors = priors.to('cuda')

        # pre = landmarks.squeeze(0)
        pre = torch.squeeze(landmarks, 0)

        tmp = (priors[:, :2] + pre[:, :2] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 2:4] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 4:6] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 6:8] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 8:10] * 0.1 * priors[:, 2:])
        landmarks = torch.cat(tmp, dim=1)
        # landmarks = landmarks * scale1
        landmarks = torch.mul(landmarks, scale1)

        landmarks = landmarks.cpu().numpy()

        # ignore low scores
        inds = torch.where(scores>score)[0]
        inds = inds.cpu().numpy()  
        scores = scores.cpu().numpy()  

        landmarks, scores = landmarks[inds], scores[inds]    

        # sort
        order = scores.argsort()[::-1]
        landmarks = landmarks[order][0]

        return np.array([[landmarks[i], landmarks[i + 1]] for i in range(0,10,2)])