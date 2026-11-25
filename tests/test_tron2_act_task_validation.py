import numpy as np

from tron2_act.task_validation import CupTaskMonitor, mask_diagnostics


def monitor():
    return CupTaskMonitor([.58,.275,.851],[.58,-.005,.851])


def tick(m,t,a,forces,support=0.,distance=.1,speeds=(0.,0.)):
    return m.update(t,.02,a,[.58,-.005,.851],[1.,1.],speeds,[0.,0.],forces,support,distance)


def test_motion_and_lift_without_grasp_do_not_pass_task():
    m=monitor()
    for k in range(100):tick(m,k*.02,[.58,-.005,.9],np.zeros(6),support=.45)
    assert not m.success
    assert 'lift' not in m.events
    assert m.report()['first_missing_stage']=='contact'


def test_palm_contact_is_not_a_finger_pinch():
    m=monitor()
    for k in range(20):tick(m,k*.02,[.58,.275,.851],[.1,0,0,0,0,.2])
    assert 'contact' in m.events
    assert 'grasp' not in m.events


def test_ordered_supported_released_stack_must_stay_stable_for_one_second():
    m=monitor()
    for k in range(10):tick(m,k*.02,[.58,.275,.851],[.2,.2,0,0,0,0])
    tick(m,.2,[.58,.275,.9],[.2,.2,0,0,0,0])
    tick(m,.22,[.58,-.005,.9],[.2,.2,0,0,0,0])
    for k in range(40):tick(m,.24+k*.02,[.58,-.005,.9],np.zeros(6),support=.45)
    assert 'released' in m.events and not m.success
    for k in range(12):tick(m,1.04+k*.02,[.58,-.005,.9],np.zeros(6),support=.45)
    assert m.success and m.report()['first_missing_stage'] is None


def test_unsettled_or_unsupported_stack_does_not_pass():
    m=monitor()
    for k in range(10):tick(m,k*.02,[.58,.275,.851],[.2,.2,0,0,0,0])
    tick(m,.2,[.58,.275,.9],[.2,.2,0,0,0,0])
    tick(m,.22,[.58,-.005,.9],[.2,.2,0,0,0,0])
    for k in range(100):tick(m,.24+k*.02,[.58,-.005,.9],np.zeros(6),support=0.)
    assert not m.success


def test_each_cup_is_scored_separately_for_missing_mask():
    semantic=np.array([[11,11,22,22],[33,33,0,0]])
    pred=np.array([[1,1,2,2],[0,0,0,0]])
    result=mask_diagnostics(pred,semantic,{'11':{'class':'hand'},'22':{'class':'cup0'},'33':{'class':'cup1'}})
    assert result['hand']['recall']==1.
    assert result['cup0']['recall']==1.
    assert result['cup1']['recall']==0.
