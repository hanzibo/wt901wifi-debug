"""Availability metrics, NOT accuracy: no labelled joint-centre ground truth."""
from collections import Counter
import numpy as np
from .rgbd_pose import NAMES


class PoseQualityStats:
    def __init__(self):
        self.frames=0;self.counts_2d=np.zeros(8,int);self.counts_3d=np.zeros(8,int)
        self.reasons=[Counter() for _ in NAMES];self.times=[]
        self.streak=np.zeros((2,8),int);self.maximum=np.zeros((2,8),int)
        self.starts=np.full((2,8),np.nan);self.longest=np.zeros((2,8))

    def break_sequence(self):
        self.streak[:]=0;self.starts[:]=np.nan

    def update(self,result,*,time_s,include_timing=True):
        self.frames+=1
        two=np.array(result.diagnostics['valid_2d'],bool);three=result.valid
        self.counts_2d+=two;self.counts_3d+=three
        for level,valid in enumerate((two,three)):
            for i,ok in enumerate(valid):
                if ok:self.streak[level,i]=0;self.starts[level,i]=np.nan
                else:
                    if self.streak[level,i]==0:self.starts[level,i]=time_s
                    self.streak[level,i]+=1
                    self.maximum[level,i]=max(self.maximum[level,i],self.streak[level,i])
                    self.longest[level,i]=max(self.longest[level,i],time_s-self.starts[level,i])
        for i,reason in enumerate(result.reasons):self.reasons[i][reason]+=1
        if include_timing:self.times.append(result.inference_ms)

    def report(self):
        parts={name:{'valid_2d_count':int(self.counts_2d[i]),'valid_3d_count':int(self.counts_3d[i]),
                'available_2d_rate':float(self.counts_2d[i]/self.frames) if self.frames else None,
                'available_3d_rate':float(self.counts_3d[i]/self.frames) if self.frames else None,
                'longest_failure_frames_2d':int(self.maximum[0,i]),'longest_failure_frames_3d':int(self.maximum[1,i]),
                'longest_failure_span_s_2d':float(self.longest[0,i]),'longest_failure_span_s_3d':float(self.longest[1,i]),
                'reasons':dict(self.reasons[i])} for i,name in enumerate(NAMES)}
        return {'frames':self.frames,'joints':parts,'timed_frames':len(self.times),
                'median_inference_ms':float(np.median(self.times)) if self.times else None,
                'p95_inference_ms':float(np.percentile(self.times,95)) if self.times else None,
                'note':'Rates use processed frames; failure span is last minus first failed sample, not exact occlusion duration. No GT accuracy.'}
