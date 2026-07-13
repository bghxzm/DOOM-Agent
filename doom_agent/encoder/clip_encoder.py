"""
Zachary Meisner
Machine Learning: Deep Learning
EN.605.740.81.SU26
Dr. Alhassan S. Yasin

Frozen OpenCLIP encoder
clip_encoder.py
"""

import numpy as np
import torch
import open_clip
from PIL import Image


class CLIP_Encoder():
    """
    OpenCLIP encoder
    """
    def __init__(self, config=None):
        self.config = config
        self.model_name = self.config['model']
        self.pretrained = self.config['pretrained']
        self.model = None
        self.preprocess = None
        self.tokenizer = None

    def init(self):
        print("\n" + "="*60)
        print(f"Model: {self.model_name} ({self.pretrained})")
        print("="*60)

        # Load model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self.model.eval()
        self.model.to(self.config['device'])
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

    def encode_frame(self, frame_array):
        '''
        Step 1: CHW -> HWC
        CHW is converted to HWC so PIL can read the array.  frame_array
        is (3, H, W) and PIL expects (H, W, 3).  Numpy's transpose function
        re-orders axes to this form.  1->0(H) 2->1(W) 0->2(C).

        Step 2: numpy -> PIL

        Step 3: PIL -> Tensor
        PIL is converted to a pre-processed tensor [3, 224, 224].
        self.preprocess resizes, center-crops, coverts to float, and normalizes.
        .unsqueeze(0) adds the batch dimension -> [1, 3, 224, 224]

        Step 4: Run the frozen encoder without gradient tracking.
        We L2 normalize because dividing the vectors magnitude makes its length
        exactly 1.  The dot product between two vectors equals their cosine
        similarity.  This matches direction instead of magnitude.  CLIP is
        trained this way so comparisons only make sense in this normalized space.

        Note: Pytorch normally builds a computation graph while running to
        calculate gradients for backprop.  Since CLIP is frozen (meaning weights
        do not get updated) we do not need this graph.  To save memory,
        and time, we use no_grad() to turn this off.

        Step 5: Remove the fake batch dimension -> [512].
        '''
        # Step 1
        hwc = frame_array.transpose(1, 2, 0)

        # Step 2
        pil_image = Image.fromarray(hwc.astype(np.uint8))

        # Step 3
        pixel_values = self.preprocess(pil_image).unsqueeze(0)
        pixel_values = pixel_values.to(self.config['device'])

        # Step 4
        with torch.no_grad():
            frame_emb = self.model.encode_image(pixel_values) # [1, 512]
            frame_emb = frame_emb / frame_emb.norm(dim=-1, keepdim=True) # L2 Normalize

        # Step 5
        return frame_emb.squeeze(0).cpu()

    def encode_subgoal(self, text):
        # CLIP's tokenizer expects a list of strings.  It converts words
        # into integer token IDs that the next transformer understands.
        tokens = self.tokenizer([text])
        tokens = tokens.to(self.config['device'])

        with torch.no_grad():
            goal_emb = self.model.encode_text(tokens) # [1, 512]
            goal_emb = goal_emb / goal_emb.norm(dim=-1, keepdim=True)

        return goal_emb.squeeze(0).cpu() # [512]
