"""Physics-only cup task scoring, never exposed to the policy observation."""
import numpy as np


class CupTaskMonitor:
    def __init__(self, initial_a, initial_b, table_height=.85):
        self.initial_a = np.asarray(initial_a, float)
        self.initial_b = np.asarray(initial_b, float)
        self.table_height = table_height
        self.events = {}
        self.contact_seconds = self.stable_seconds = 0.
        self.max_lift = self.max_hand_contact_n = 0.
        self.min_alignment = float('inf')
        self.last = None
        self.success = False

    def update(self, t, dt, a, b, upright, linear_speeds, angular_speeds,
               hand_forces, support_force, hand_distance):
        a,b = np.asarray(a),np.asarray(b)
        force = np.asarray(hand_forces, float)
        if force.ndim != 1 or len(force) < 2 or not np.isfinite(force).all():
            raise ValueError('Need measured thumb and finger contact forces')
        contact = force.sum()
        lift = float(a[2]-self.initial_a[2])
        alignment = float(np.linalg.norm(a[:2]-b[:2]))
        self.max_lift = max(self.max_lift,lift)
        self.max_hand_contact_n = max(self.max_hand_contact_n,float(contact))
        self.min_alignment = min(self.min_alignment,alignment)
        if contact > .02:
            self.events.setdefault('contact',t)
        pinch = force[0]>.02 and np.any(force[1:5]>.02)
        self.contact_seconds = self.contact_seconds+dt if pinch else 0.
        if self.contact_seconds >= .15:
            self.events.setdefault('grasp',t)
        if 'grasp' in self.events and lift>.03 and contact>.02:
            self.events.setdefault('lift',t)
        nested = (alignment<.008 and .015<a[2]-b[2]<.085 and min(upright)>.97
                  and abs(b[2]-self.initial_b[2])<.01
                  and np.linalg.norm(b[:2]-self.initial_b[:2])<.015)
        if 'lift' in self.events and nested:
            self.events.setdefault('aligned',t)
        released = contact<.03 and hand_distance>.06
        if 'aligned' in self.events and released:
            self.events.setdefault('released',t)
        stable = (nested and released and support_force>.02
                  and max(linear_speeds)<.01 and max(angular_speeds)<.15)
        self.stable_seconds = self.stable_seconds+dt if stable and 'released' in self.events else 0.
        if self.stable_seconds >= 1.:
            self.events.setdefault('stable_stack',t)
            self.success = True
        self.last = {'t':float(t),'lift_m':lift,'alignment_m':alignment,
                     'hand_contact_n':float(contact),'support_contact_n':float(support_force),
                     'pinch':bool(pinch),'nested':bool(nested),'released':bool(released),
                     'stable_s':self.stable_seconds}
        return self.last

    def report(self):
        order = ['contact','grasp','lift','aligned','released','stable_stack']
        return {'success':self.success,'events_s':self.events,
                'first_missing_stage':next((x for x in order if x not in self.events),None),
                'max_lift_m':self.max_lift,'max_hand_contact_n':self.max_hand_contact_n,
                'min_alignment_m':None if not np.isfinite(self.min_alignment) else self.min_alignment,
                'final':self.last,
                'criteria':'thumb+finger contact >=0.15s; lift >30mm; nested XY<8mm; release; real cup support contact and low velocities for >=1s'}


def mask_diagnostics(prediction, semantic, labels):
    """Score the frozen segmenter with simulator labels; labels are diagnostics only."""
    result = {}
    for name in ['hand','cup0','cup1']:
        ids = [int(k) for k,v in labels.items() if v.get('class') == name]
        gt = np.isin(semantic,ids)
        pred = prediction == (1 if name=='hand' else 2)
        visible = int(gt.sum())
        result[name] = {'visible_pixels':visible,
            'recall':None if not visible else float((gt&pred).sum()/visible)}
    return result
