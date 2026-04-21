class Config:
    IMG_SIZE = 256

    STEM_OUT = 32

    STAGE_CHANNELS = [
        [40, 80],            
        [40, 80, 160],        
        [40, 80, 160, 320]      
    ]

    STAGE_REPEATS = [2, 4, 2]

    NUM_KEYPOINTS = 17
