"""Analytic geometry, coordinate invariance, and endpoint consistency checks."""
import numpy as np
from final_metrics import features,histograms,contacts,QUEUE,load,score,freeze_reference,executed
# Constant acceleration remains longitudinal after a rigid rotation.
t=np.arange(101)*.1;x=np.c_[t*t,np.zeros_like(t)];av=np.ones(101,bool)
for angle in [0.,np.pi/2,.7]:
 r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
 f=features(x@r.T+np.array([123.,-87.]),np.full(101,angle),av)
 np.testing.assert_allclose(f['lon_accel'],2.,atol=1e-10)
 np.testing.assert_allclose(f['lat_accel'],0.,atol=1e-10)
 np.testing.assert_allclose(f['jerk'],0.,atol=1e-9)
h=histograms(dict(lon_accel=np.array([-200.,0.,200.]),lat_accel=np.array([0.]),jerk=np.array([0.])))
assert sum(h['lon_accel']['counts'])==3 and h['lon_accel']['below']==h['lon_accel']['above']==1
# Two 4-by-2 rectangles touch exactly at the final timestamp.
xy=np.array([[[0.,0.],[0.,0.],[0.,0.]],[[6.,0.],[5.,0.],[4.,0.]]]);yaw=np.zeros((2,3));ext=np.tile([4.,2.,1.],(2,3,1))
_,hit=contacts(xy,yaw,ext,np.ones((2,3),bool));assert hit[0].tolist()==[False,False,True]
protocol=freeze_reference()
# This real trajectory previously failed the 1 cm alignment check because its
# exported inverse matrix had 3.9 cm of translation error. Reconstructing with
# the simulator's centroid/yaw must agree without weakening the tolerance.
regression=QUEUE/'runs'/'scene-0104_s1_idm_A2'
path,b,_,_,_,_,alignment=executed(regression)
assert alignment<.001,alignment
import h5py
with h5py.File(path) as h:
 g=h[next(iter(h))];ids=[b['runtime_ego_index'],b['runtime_target_index']]
 xy=np.asarray(g['centroid'],float)[ids];tf=np.asarray(g['world_from_agent'],float)[ids];local=np.asarray(g['action_positions'],float)[ids,:,0]
 origin=tf[:,(np.arange(100)//5)*5]
 old_future=np.einsum('ntij,ntj->nti',origin[:,:,:2,:2],local)+origin[:,:,:2,2]
 old_error=float(np.linalg.norm(old_future[:,:-1]-xy[:,1:],axis=-1).max())
assert old_error>.01,old_error
print('TRANSFORM_REGRESSION',dict(stored_matrix_error_m=old_error,simulator_transform_error_m=alignment))
for jobid in ['scene-1061_s1_idm_A0']:
 d=QUEUE/'runs'/jobid;m=score(d,protocol)
 assert m['ADE']==m['Adv_ADE'] and m['FDE']==m['Adv_FDE']
 # This fixture's logged target exits before the fixed 10 s endpoint.
 # Its valid prefix contributes ADE; FDE must remain missing, never use a
 # shorter endpoint or zero. The generated target still spans all 101 states.
 assert m['valid_paired_frames']==54 and m['valid_target_frames']==101
 assert m['FDE'] is None and m['valid_FDE_trajectories']==0
 assert (m['MS'] is None)==(not m['CR'])
 assert m['action_frame_alignment_max_m']<.01
 print(jobid,{k:m[k] for k in ['ADE','FDE','RM','CR','MS','action_frame_alignment_max_m']})
print('ANALYTIC_AND_REAL_ARTIFACT_CHECKS_PASSED')
