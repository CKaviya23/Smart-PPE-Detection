"""
app.py — Safety PPE Detection System v2 (with SQLite Analytics)
Model: D:/safety_ppe/ppe_yolov8s_run/weights/best.pt
Classes (14):
  0:Fall-Detected  1:Gloves       2:Goggles     3:Hardhat
  4:Ladder         5:Mask         6:NO-Gloves   7:NO-Goggles
  8:NO-Hardhat     9:NO-Mask     10:NO-Safety Vest  11:Person
  12:Safety Cone  13:Safety Vest

Run: python app.py | Open: http://127.0.0.1:5000
Analytics: http://127.0.0.1:5000/analytics
"""

import os, sys, cv2, base64, threading, traceback, sqlite3, json
from datetime import datetime
import numpy as np
from collections import OrderedDict
from flask import Flask, render_template, request, jsonify, Response, stream_with_context

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: Run: pip install ultralytics")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
MODEL_PATH       = r"D:\safety_ppe\ppe_yolov8s_run\weights\best.pt"
DB_PATH          = "ppe_detections.db"
CONF_THRESHOLD   = 0.30
NMS_THRESHOLD    = 0.45
UPLOAD_FOLDER    = os.path.join("static", "uploads")
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
ALLOWED_IMG      = {"png", "jpg", "jpeg", "bmp", "webp"}
ALLOWED_VID      = {"mp4", "avi", "mov", "mkv", "webm"}
TRIPWIRE_X_RATIO = 0.50
BLUR_THRESH      = 80.0
BRIGHT_LOW       = 20
BRIGHT_HIGH      = 240
CHANGE_THRESH    = 50.0

# ══════════════════════════════════════════════════════════════
#  CLASS IDs
# ══════════════════════════════════════════════════════════════
CLS_FALL=0; CLS_GLOVES=1; CLS_GOGGLES=2; CLS_HARDHAT=3
CLS_LADDER=4; CLS_MASK=5; CLS_NO_GLOVES=6; CLS_NO_GOGGLES=7
CLS_NO_HARDHAT=8; CLS_NO_MASK=9; CLS_NO_VEST=10; CLS_PERSON=11
CLS_CONE=12; CLS_VEST=13

PPE_CLASS_IDS = {
    "all"    : [0,1,2,3,5,6,7,8,9,10,11,13],
    "helmet" : [CLS_HARDHAT, CLS_NO_HARDHAT],
    "glasses": [CLS_GOGGLES, CLS_NO_GOGGLES],
    "gloves" : [CLS_GLOVES,  CLS_NO_GLOVES],
    "vest"   : [CLS_VEST,    CLS_NO_VEST],
    "mask"   : [CLS_MASK,    CLS_NO_MASK],
    "person" : [CLS_PERSON],
}

VIOLATION_IDS = {CLS_FALL,CLS_NO_GLOVES,CLS_NO_GOGGLES,
                 CLS_NO_HARDHAT,CLS_NO_MASK,CLS_NO_VEST}

CLASS_COLOURS = {
    CLS_FALL:(0,0,220), CLS_GLOVES:(50,200,180), CLS_GOGGLES:(180,80,220),
    CLS_HARDHAT:(50,220,50), CLS_LADDER:(200,200,0), CLS_MASK:(220,120,50),
    CLS_NO_GLOVES:(0,0,220), CLS_NO_GOGGLES:(0,0,220), CLS_NO_HARDHAT:(0,0,220),
    CLS_NO_MASK:(0,0,220), CLS_NO_VEST:(0,0,220), CLS_PERSON:(255,200,0),
    CLS_CONE:(0,140,255), CLS_VEST:(0,200,255),
}

_model=None; _classes={}; _person_model=None
_det_cache=[]; _CACHE_FRAMES=3
_vest_memory={}; _VEST_HOLD=8

# ══════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════
def init_db():
    """Create all required tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Session table — one row per detection run
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT    NOT NULL,
        task        TEXT    NOT NULL,
        source      TEXT    NOT NULL,
        duration_ms INTEGER DEFAULT 0
    )''')

    # Task 1: Person detections
    c.execute('''CREATE TABLE IF NOT EXISTS person_detections (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  INTEGER,
        timestamp   TEXT,
        person_id   INTEGER,
        confidence  REAL,
        FOREIGN KEY(session_id) REFERENCES sessions(id)
    )''')

    # Task 2: PPE detections
    c.execute('''CREATE TABLE IF NOT EXISTS ppe_detections (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  INTEGER,
        timestamp   TEXT,
        class_name  TEXT,
        confidence  REAL,
        is_violation INTEGER,
        FOREIGN KEY(session_id) REFERENCES sessions(id)
    )''')

    # Task 3: Tripwire events
    c.execute('''CREATE TABLE IF NOT EXISTS tripwire_events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  INTEGER,
        timestamp   TEXT,
        breach      INTEGER,
        persons     INTEGER,
        breach_count INTEGER,
        FOREIGN KEY(session_id) REFERENCES sessions(id)
    )''')

    # Task 4: Tamper events
    c.execute('''CREATE TABLE IF NOT EXISTS tamper_events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  INTEGER,
        timestamp   TEXT,
        tampered    INTEGER,
        reasons     TEXT,
        brightness  REAL,
        motion      REAL,
        FOREIGN KEY(session_id) REFERENCES sessions(id)
    )''')

    conn.commit()
    conn.close()
    print("[DB] Database initialized ✓")


def db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def log_session(task, source):
    """Create a new session and return its ID."""
    conn = db_conn()
    c = conn.cursor()
    c.execute("INSERT INTO sessions(timestamp,task,source) VALUES(?,?,?)",
              (datetime.now().isoformat(), task, source))
    sid = c.lastrowid
    conn.commit(); conn.close()
    return sid


def log_person(session_id, person_id, confidence):
    conn = db_conn()
    conn.execute("INSERT INTO person_detections(session_id,timestamp,person_id,confidence) VALUES(?,?,?,?)",
                 (session_id, datetime.now().isoformat(), person_id, confidence))
    conn.commit(); conn.close()


def log_ppe(session_id, detections):
    """detections: list of {class, confidence, violation}"""
    if not detections: return
    conn = db_conn()
    ts = datetime.now().isoformat()
    for d in detections:
        conn.execute("INSERT INTO ppe_detections(session_id,timestamp,class_name,confidence,is_violation) VALUES(?,?,?,?,?)",
                     (session_id, ts, d.get("class",""), d.get("confidence",0), int(d.get("violation",False))))
    conn.commit(); conn.close()


def log_tripwire(session_id, breach, persons, breach_count):
    conn = db_conn()
    conn.execute("INSERT INTO tripwire_events(session_id,timestamp,breach,persons,breach_count) VALUES(?,?,?,?,?)",
                 (session_id, datetime.now().isoformat(), int(breach), persons, breach_count))
    conn.commit(); conn.close()


def log_tamper(session_id, tampered, reasons, brightness, motion):
    conn = db_conn()
    conn.execute("INSERT INTO tamper_events(session_id,timestamp,tampered,reasons,brightness,motion) VALUES(?,?,?,?,?,?)",
                 (session_id, datetime.now().isoformat(), int(tampered), json.dumps(reasons), brightness, motion))
    conn.commit(); conn.close()

# ══════════════════════════════════════════════════════════════
#  MODEL LOADER
# ══════════════════════════════════════════════════════════════
def load_model():
    global _model, _classes, _person_model
    print(f"\n[Model] Loading: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"[Model] ERROR: File not found: {MODEL_PATH}")
        sys.exit(1)
    _model   = YOLO(MODEL_PATH)
    _classes = _model.names
    print(f"[Model] Classes: {_classes}")
    global CLS_PERSON
    for cid, cname in _classes.items():
        if cname.lower() in ["person","worker","human","people"]:
            CLS_PERSON = cid
            print(f"[Model] Person class: ID={cid} name={cname}")
            break
    print("[Model] Loading yolov8n for person detection...")
    _person_model = YOLO("yolov8n.pt")
    print("[Model] Person model ready ✓\n")

# ══════════════════════════════════════════════════════════════
#  CENTROID TRACKER
# ══════════════════════════════════════════════════════════════
class CentroidTracker:
    def __init__(self, max_gone=60, max_dist=80):
        self.nid=0; self.objs=OrderedDict(); self.gone=OrderedDict()
        self.max_gone=max_gone; self.max_dist=max_dist
    def reset(self): self.__init__(self.max_gone, self.max_dist)
    def update(self, rects):
        if not rects:
            for oid in list(self.gone):
                self.gone[oid]+=1
                if self.gone[oid]>self.max_gone:
                    del self.objs[oid]; del self.gone[oid]
            return self.objs
        nc=np.array([((x1+x2)//2,(y1+y2)//2) for x1,y1,x2,y2 in rects])
        if not self.objs:
            for c in nc:
                self.objs[self.nid]=tuple(c); self.gone[self.nid]=0; self.nid+=1
        else:
            ids=list(self.objs.keys())
            old=np.array(list(self.objs.values()))
            D=np.linalg.norm(old[:,None]-nc[None,:],axis=2)
            rows=D.min(axis=1).argsort(); cols=D.argmin(axis=1)[rows]
            ur,uc=set(),set()
            for r,c in zip(rows,cols):
                if r in ur or c in uc: continue
                if D[r,c]>self.max_dist: continue
                oid=ids[r]; self.objs[oid]=tuple(nc[c]); self.gone[oid]=0
                ur.add(r); uc.add(c)
            for r in set(range(len(ids)))-ur:
                oid=ids[r]; self.gone[oid]+=1
                if self.gone[oid]>self.max_gone:
                    del self.objs[oid]; del self.gone[oid]
            for c in set(range(len(nc)))-uc:
                self.objs[self.nid]=tuple(nc[c]); self.gone[self.nid]=0; self.nid+=1
        return self.objs

_tracker=CentroidTracker()

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def get_colour(cid): return CLASS_COLOURS.get(cid,(180,180,180))

def draw_box(frame,x1,y1,x2,y2,label,cid):
    col=get_colour(cid); cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
    (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.55,1)
    cv2.rectangle(frame,(x1,y1-th-6),(x1+tw+6,y1),col,-1)
    cv2.putText(frame,label,(x1+3,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),1,cv2.LINE_AA)

def draw_status(frame,text,ok=True):
    h,w=frame.shape[:2]; cv2.rectangle(frame,(0,h-30),(w,h),(15,15,15),-1)
    cv2.putText(frame,text,(8,h-9),cv2.FONT_HERSHEY_SIMPLEX,0.55,
                (0,220,60) if ok else (30,30,220),1,cv2.LINE_AA)

def to_b64(frame):
    _,buf=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,85])
    return base64.b64encode(buf).decode()

def bytes_to_frame(data):
    return cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)

def infer(frame,class_ids=None):
    res=_model(frame,conf=CONF_THRESHOLD,iou=NMS_THRESHOLD,classes=class_ids,imgsz=640,verbose=False)
    dets=[]
    for r in res:
        for box in r.boxes:
            cid=int(box.cls[0]); cname=_classes.get(cid,f"cls{cid}")
            conf=float(box.conf[0]); x1,y1,x2,y2=map(int,box.xyxy[0])
            dets.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"conf":conf,"cid":cid,"cname":cname})
    hard_cls=[CLS_VEST,CLS_NO_VEST,CLS_GLOVES,CLS_NO_GLOVES,CLS_GOGGLES,CLS_NO_GOGGLES,CLS_MASK,CLS_NO_MASK]
    if class_ids is None or any(c in (class_ids or []) for c in hard_cls):
        res2=_model(frame,conf=0.05,iou=NMS_THRESHOLD,classes=hard_cls,imgsz=640,verbose=False)
        for r in res2:
            for box in r.boxes:
                cid=int(box.cls[0]); cname=_classes.get(cid,f"cls{cid}")
                conf=float(box.conf[0]); x1,y1,x2,y2=map(int,box.xyxy[0])
                dets.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"conf":conf,"cid":cid,"cname":cname})
    return deduplicate(dets)

def deduplicate(dets,thresh=0.50):
    if len(dets)<=1: return dets
    keep=[]; used=set()
    for i,a in enumerate(dets):
        if i in used: continue
        for j,b in enumerate(dets):
            if j<=i or j in used: continue
            if a["cid"]==b["cid"] and box_iou(a,b)>thresh:
                if b["conf"]>a["conf"]: used.add(i)
                else: used.add(j)
        if i not in used: keep.append(a)
    return keep

def box_iou(a,b):
    xi1=max(a["x1"],b["x1"]); yi1=max(a["y1"],b["y1"])
    xi2=min(a["x2"],b["x2"]); yi2=min(a["y2"],b["y2"])
    inter=max(0,xi2-xi1)*max(0,yi2-yi1)
    ua=((a["x2"]-a["x1"])*(a["y2"]-a["y1"])+(b["x2"]-b["x1"])*(b["y2"]-b["y1"])-inter)
    return inter/ua if ua>0 else 0

def _nms_persons(dets,iou_thresh=0.40):
    if len(dets)<=1: return dets
    dets=sorted(dets,key=lambda x:x["conf"],reverse=True)
    keep=[]; used=set()
    for i,a in enumerate(dets):
        if i in used: continue
        keep.append(a)
        for j,b in enumerate(dets):
            if j<=i or j in used: continue
            xi1=max(a["x1"],b["x1"]); yi1=max(a["y1"],b["y1"])
            xi2=min(a["x2"],b["x2"]); yi2=min(a["y2"],b["y2"])
            inter=max(0,xi2-xi1)*max(0,yi2-yi1)
            ua=((a["x2"]-a["x1"])*(a["y2"]-a["y1"])+(b["x2"]-b["x1"])*(b["y2"]-b["y1"])-inter)
            if inter/(ua if ua>0 else 1)>iou_thresh: used.add(j)
    return keep

# ══════════════════════════════════════════════════════════════
#  DETECTION TASKS
# ══════════════════════════════════════════════════════════════

# ── Task 1: Person + ID ──────────────────────────────────────
def detect_persons(frame, reset=False, session_id=None):
    global _tracker
    if reset: _tracker.reset()
    res=_person_model(frame,conf=0.35,iou=0.50,classes=[0],verbose=False)
    dets=[]
    for r in res:
        for box in r.boxes:
            conf=float(box.conf[0]); x1,y1,x2,y2=map(int,box.xyxy[0])
            dets.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"conf":conf})
    dets=_nms_persons(dets,iou_thresh=0.40)
    rects=[(d["x1"],d["y1"],d["x2"],d["y2"]) for d in dets]
    tracked=_tracker.update(rects)
    results=[]
    for i,d in enumerate(dets):
        cx=(d["x1"]+d["x2"])//2; cy=(d["y1"]+d["y2"])//2
        best_id=i; best_dist=float("inf")
        for oid,c in tracked.items():
            dist=abs(c[0]-cx)+abs(c[1]-cy)
            if dist<best_dist: best_dist=dist; best_id=oid
        draw_box(frame,d["x1"],d["y1"],d["x2"],d["y2"],
                 f"Person ID:{best_id}  {d['conf']:.0%}",CLS_PERSON)
        results.append({"id":int(best_id),"confidence":round(d["conf"],3)})
        if session_id:
            log_person(session_id, int(best_id), round(d["conf"],3))
    draw_status(frame,f"Persons Detected: {len(dets)}")
    return frame, results


# ── Task 2: PPE Detection ────────────────────────────────────
def detect_ppe(frame, selected_types=None, session_id=None):
    global _vest_memory
    if not selected_types or "all" in selected_types:
        class_ids=PPE_CLASS_IDS["all"]; tag="ALL"; check_vest=True
    else:
        class_ids=[]
        for t in selected_types: class_ids.extend(PPE_CLASS_IDS.get(t,[]))
        class_ids=list(set(class_ids)); tag="+".join(selected_types).upper()
        check_vest=any(t in ["vest","all"] for t in (selected_types or []))
    dets=infer(frame,class_ids=class_ids if class_ids else None)
    ok=0; bad=0; results=[]; vest_dets=[]
    for d in dets:
        cid=d["cid"]; cname=d["cname"]
        if cid in [CLS_VEST,CLS_NO_VEST]: vest_dets.append(d); continue
        label=f"{cname.upper()}  {d['conf']:.0%}"
        draw_box(frame,d["x1"],d["y1"],d["x2"],d["y2"],label,cid)
        v=cid in VIOLATION_IDS
        if v: bad+=1
        else: ok+=1
        results.append({"class":cname,"confidence":round(d["conf"],3),"violation":v})
    if check_vest:
        p_res=_person_model(frame,conf=0.40,iou=0.45,classes=[0],verbose=False)
        persons=[]
        for r in p_res:
            for box in r.boxes:
                x1,y1,x2,y2=map(int,box.xyxy[0])
                if (x2-x1)<40 or (y2-y1)<60: continue
                persons.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"conf":float(box.conf[0])})
        persons=_nms_persons(persons,iou_thresh=0.45)
        _vest_memory={k:v for k,v in _vest_memory.items() if k<len(persons)}
        for i,p in enumerate(persons):
            px1,py1,px2,py2=p["x1"],p["y1"],p["x2"],p["y2"]; ph=py2-py1
            torso_y1=py1+int(ph*0.35); torso_y2=py1+int(ph*0.78)
            best_vest=None; best_conf=0.0
            for v in vest_dets:
                vcx=(v["x1"]+v["x2"])//2; vcy=(v["y1"]+v["y2"])//2
                if px1<=vcx<=px2 and torso_y1<=vcy<=torso_y2:
                    if v["conf"]>best_conf: best_conf=v["conf"]; best_vest=v
            mem=_vest_memory.get(i,{"has_vest":False,"hold":0})
            if best_vest is not None: mem["has_vest"]=True; mem["hold"]=_VEST_HOLD
            else:
                if mem["hold"]>0: mem["hold"]-=1
                else: mem["has_vest"]=False
            _vest_memory[i]=mem
            if mem["has_vest"]:
                col=CLASS_COLOURS[CLS_VEST]; tx1,ty1,tx2,ty2=px1,torso_y1,px2,torso_y2
                cv2.rectangle(frame,(tx1,ty1),(tx2,ty2),col,2)
                lbl=f"SAFETY VEST  {best_conf:.0%}" if best_vest else "SAFETY VEST"
                (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.52,1)
                cv2.rectangle(frame,(tx1,ty1-th-5),(tx1+tw+4,ty1),col,-1)
                cv2.putText(frame,lbl,(tx1+2,ty1-3),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,0,0),1,cv2.LINE_AA)
                ok+=1; results.append({"class":"Safety Vest","confidence":round(best_conf,3),"violation":False})
            else:
                col=(0,0,220); tx1,ty1,tx2,ty2=px1,torso_y1,px2,torso_y2
                cv2.rectangle(frame,(tx1,ty1),(tx2,ty2),col,2)
                lbl="NO-SAFETY VEST"
                (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.52,1)
                cv2.rectangle(frame,(tx1,ty1-th-5),(tx1+tw+4,ty1),col,-1)
                cv2.putText(frame,lbl,(tx1+2,ty1-3),cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),1,cv2.LINE_AA)
                bad+=1; results.append({"class":"NO-Safety Vest","confidence":1.0,"violation":True})
    else:
        for d in vest_dets:
            cid=d["cid"]; cname=d["cname"]
            draw_box(frame,d["x1"],d["y1"],d["x2"],d["y2"],f"{cname.upper()}  {d['conf']:.0%}",cid)
            v=cid in VIOLATION_IDS
            if v: bad+=1
            else: ok+=1
            results.append({"class":cname,"confidence":round(d["conf"],3),"violation":v})
    if session_id and results:
        log_ppe(session_id, results)
    draw_status(frame,f"PPE [{tag}]  OK:{ok}  Violations:{bad}",ok=(bad==0))
    return frame, results


# ── Task 3: Polygon Zone Tripwire ────────────────────────────
_tripwire_zone=[]; _breach_count=0; _breach_flash=0; _person_in_zone={}

def set_tripwire_zone(points_pct):
    global _tripwire_zone,_breach_count,_breach_flash,_person_in_zone
    _tripwire_zone=points_pct; _breach_count=0; _breach_flash=0; _person_in_zone={}

def point_in_polygon(px,py,polygon):
    n=len(polygon)
    if n<3: return False
    inside=False; j=n-1
    for i in range(n):
        xi,yi=polygon[i]; xj,yj=polygon[j]
        if ((yi>py)!=(yj>py)) and (px<(xj-xi)*(py-yi)/(yj-yi+1e-9)+xi):
            inside=not inside
        j=i
    return inside

def detect_tripwire(frame, reset=False, session_id=None):
    global _breach_count,_breach_flash,_person_in_zone
    if reset: _breach_count=0; _breach_flash=0; _person_in_zone={}
    h,w=frame.shape[:2]
    if len(_tripwire_zone)<3:
        cv2.putText(frame,"Draw a zone in the UI to start detection",
                    (20,h//2),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,220,220),2,cv2.LINE_AA)
        draw_status(frame,"Tripwire: Draw zone first — click on image to add points")
        return frame,{"breach":False,"breach_count":0,"persons":0}
    zone_px=[(int(p[0]*w),int(p[1]*h)) for p in _tripwire_zone]
    res_p=_person_model(frame,conf=0.30,iou=0.45,classes=[0],imgsz=416,verbose=False)
    dets=[]
    for r in res_p:
        for box in r.boxes:
            conf=float(box.conf[0]); x1,y1,x2,y2=map(int,box.xyxy[0])
            dets.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"conf":conf})
    dets=_nms_persons(dets,iou_thresh=0.40)
    new_states={}
    for i,d in enumerate(dets):
        foot_x=(d["x1"]+d["x2"])//2; foot_y=d["y2"]
        cx=(d["x1"]+d["x2"])//2; cy=(d["y1"]+d["y2"])//2
        best_key=None; best_dist=999999
        for key,state in _person_in_zone.items():
            dist=abs(state.get("cx",0)-cx)+abs(state.get("cy",0)-cy)
            if dist<best_dist and dist<120: best_dist=dist; best_key=key
        prev_state=_person_in_zone.get(best_key,{"inside":False,"cooldown":0,"cx":cx,"cy":cy})
        cx_body=(d["x1"]+d["x2"])//2; cy_body=(d["y1"]+d["y2"])//2
        chest_y=d["y1"]+(d["y2"]-d["y1"])//3
        inside=(point_in_polygon(foot_x,foot_y,zone_px) or
                point_in_polygon(cx_body,cy_body,zone_px) or
                point_in_polygon(cx_body,chest_y,zone_px) or
                point_in_polygon(d["x1"]+10,cy_body,zone_px) or
                point_in_polygon(d["x2"]-10,cy_body,zone_px))
        cooldown=prev_state.get("cooldown",0)
        if inside:
            if not prev_state.get("inside",False) and cooldown==0:
                _breach_count+=1; cooldown=45
            _breach_flash=20
        if cooldown>0: cooldown-=1
        new_states[i]={"inside":inside,"cooldown":cooldown,"cx":cx,"cy":cy}
        col=(0,0,220) if inside else (255,200,0)
        cv2.rectangle(frame,(d["x1"],d["y1"]),(d["x2"],d["y2"]),col,2)
        lbl=f"{'!! INTRUDER !!' if inside else 'Person'}  {d['conf']:.0%}"
        (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.55,1)
        cv2.rectangle(frame,(d["x1"],d["y1"]-th-6),(d["x1"]+tw+6,d["y1"]),col,-1)
        cv2.putText(frame,lbl,(d["x1"]+3,d["y1"]-4),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),1,cv2.LINE_AA)
        cv2.circle(frame,(foot_x,foot_y),6,col,-1)
    _person_in_zone=new_states
    if _breach_flash>0: _breach_flash-=1
    active_breach=_breach_flash>0
    zone_arr=np.array(zone_px,dtype=np.int32)
    zone_col=(0,0,220) if active_breach else (0,220,220)
    overlay=frame.copy(); fill_col=(0,0,120) if active_breach else (0,60,60)
    cv2.fillPoly(overlay,[zone_arr],fill_col)
    frame[:]=cv2.addWeighted(overlay,0.30,frame,0.70,0)
    cv2.polylines(frame,[zone_arr],True,zone_col,3)
    zx=int(np.mean([p[0] for p in zone_px])); zy=int(np.mean([p[1] for p in zone_px]))
    cv2.putText(frame,"RESTRICTED ZONE",(zx-75,zy),cv2.FONT_HERSHEY_SIMPLEX,0.65,zone_col,2,cv2.LINE_AA)
    if active_breach:
        banner=frame.copy()
        cv2.rectangle(banner,(0,0),(w,70),(0,0,160),-1)
        frame[:]=cv2.addWeighted(banner,0.65,frame,0.35,0)
        cv2.putText(frame,f"!! ZONE BREACH ALERT !!  Total: {_breach_count}",
                    (10,48),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,50,255),3,cv2.LINE_AA)
    draw_status(frame,f"Tripwire: {'BREACH ALERT!' if active_breach else 'OK'}  Persons:{len(dets)}  Breaches:{_breach_count}",ok=not active_breach)
    result={"breach":active_breach,"breach_count":_breach_count,"persons":len(dets)}
    if session_id:
        log_tripwire(session_id, active_breach, len(dets), _breach_count)
    return frame, result


# ── Task 4: Camera Tampering ─────────────────────────────────
_prev_gray=None; _bright_history=[]; _motion_buf=[]; _tamper_alert_hold=0; _warmup_frames=0

def detect_tampering(frame, reset=False, session_id=None):
    global _prev_gray,_bright_history,_motion_buf,_tamper_alert_hold,_warmup_frames
    if reset:
        _prev_gray=None; _bright_history=[]; _motion_buf=[]; _tamper_alert_hold=0; _warmup_frames=0
    h,w=frame.shape[:2]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); gray_b=cv2.GaussianBlur(gray,(9,9),0)
    bright=float(np.mean(gray_b))
    if _prev_gray is not None and _prev_gray.shape==gray_b.shape:
        motion=float(np.mean(cv2.absdiff(gray_b,_prev_gray)))
    else: motion=0.0
    _prev_gray=gray_b.copy()
    _warmup_frames+=1
    if _warmup_frames<=30:
        _bright_history.append(bright)
        draw_status(frame,f"Camera: Starting ({_warmup_frames}/30)...")
        return frame,{"tampered":False,"reasons":[],"brightness":round(bright,1),"motion":0}
    _bright_history.append(bright)
    if len(_bright_history)>60: _bright_history.pop(0)
    baseline=float(np.median(_bright_history)); sudden_change=abs(bright-baseline)
    _motion_buf.append(motion)
    if len(_motion_buf)>6: _motion_buf.pop(0)
    peak_motion=max(_motion_buf)
    reasons=[]
    if sudden_change>40: reasons.append("Camera Covered")
    if peak_motion>12.0: reasons.append("Camera Shaking")
    is_tampered=bool(reasons)
    if is_tampered: _tamper_alert_hold=40
    elif _tamper_alert_hold>0: _tamper_alert_hold-=1
    show_alert=_tamper_alert_hold>0
    if show_alert:
        ov=frame.copy(); cv2.rectangle(ov,(0,0),(w,h),(0,0,150),-1)
        frame=cv2.addWeighted(ov,0.35,frame,0.65,0)
        cv2.rectangle(frame,(3,3),(w-3,h-3),(0,0,255),4)
        cv2.putText(frame,"!! CAMERA TAMPERED !!",(w//2-200,h//2-20),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),3,cv2.LINE_AA)
        show_r=reasons if reasons else ["Camera tampered"]
        for idx,r in enumerate(show_r[:2]):
            cv2.putText(frame,r,(w//2-200,h//2+25+idx*30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)
    else:
        cv2.putText(frame,f"Bright:{bright:.0f} Base:{baseline:.0f} Diff:{sudden_change:.0f} Motion:{peak_motion:.1f}",
                    (8,22),cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,200,60),1,cv2.LINE_AA)
        cv2.putText(frame,"Camera: OK",(8,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,60),1,cv2.LINE_AA)
    draw_status(frame,f"Tamper: {'ALERT! '+reasons[0] if show_alert and reasons else 'OK'}",ok=not show_alert)
    result={"tampered":show_alert,"reasons":reasons,"brightness":round(bright,1),"motion":round(peak_motion,1)}
    if session_id and (show_alert or _warmup_frames % 30 == 0):
        log_tamper(session_id, show_alert, reasons, round(bright,1), round(peak_motion,1))
    return frame, result


def dispatch(frame, task, selected_ppe=None, reset=False, session_id=None):
    try:
        if   task=="person":   return detect_persons(frame, reset=reset, session_id=session_id)
        elif task=="tripwire": return detect_tripwire(frame, reset=reset, session_id=session_id)
        elif task=="tamper":   return detect_tampering(frame, reset=reset, session_id=session_id)
        else:                  return detect_ppe(frame, selected_types=selected_ppe or ["all"], session_id=session_id)
    except Exception as e:
        traceback.print_exc()
        draw_status(frame,f"Error: {e}",ok=False)
        return frame,{"error":str(e)}

# ══════════════════════════════════════════════════════════════
#  FLASK
# ══════════════════════════════════════════════════════════════
app=Flask(__name__)
app.config["MAX_CONTENT_LENGTH"]=MAX_UPLOAD_BYTES
os.makedirs(UPLOAD_FOLDER,exist_ok=True)
_cam_lock=threading.Lock(); _cam=None; _cam_task="ppe"; _cam_ppe=["all"]
_current_session_id=None

def ext_ok(fn,al): return "."+fn.rsplit(".",1)[-1].lower() in {"."+x for x in al}
def parse_ppe(v):
    if not v: return ["all"]
    p=[x.strip() for x in v.split(",") if x.strip()]
    return p if p else ["all"]

@app.route("/")
def index(): return render_template("index.html")

@app.route("/analytics")
def analytics(): return render_template("analytics.html")

@app.route("/api/tripwire/zone",methods=["POST"])
def set_zone():
    data=request.get_json(silent=True) or {}
    points=data.get("points",[])
    pts=[(p["x"],p["y"]) for p in points]
    set_tripwire_zone(pts)
    return jsonify(ok=True,points=len(pts))

@app.route("/api/photo",methods=["POST"])
def api_photo():
    if "file" not in request.files: return jsonify(error="No file"),400
    f=request.files["file"]
    if not ext_ok(f.filename,ALLOWED_IMG): return jsonify(error="Bad format"),400
    frame=bytes_to_frame(f.read())
    if frame is None: return jsonify(error="Cannot decode"),400
    task=request.form.get("task","ppe")
    sel=parse_ppe(request.form.get("selected_ppe","all"))
    global _det_cache,_vest_memory
    _det_cache=[]; _vest_memory={}
    sid=log_session(task,"photo")
    ann,res=dispatch(frame,task,selected_ppe=sel,reset=True,session_id=sid)
    return jsonify(image=to_b64(ann),results=res,session_id=sid)

@app.route("/api/video",methods=["POST"])
def api_video():
    if "file" not in request.files: return jsonify(error="No file"),400
    f=request.files["file"]
    if not ext_ok(f.filename,ALLOWED_VID): return jsonify(error="Bad format"),400
    task=request.form.get("task","ppe")
    sel=parse_ppe(request.form.get("selected_ppe","all"))
    tmp=os.path.join(UPLOAD_FOLDER,"upload_video.mp4"); f.save(tmp)
    global _det_cache,_vest_memory
    _det_cache=[]; _vest_memory={}
    sid=log_session(task,"video")
    def gen():
        cap=cv2.VideoCapture(tmp); first=True
        while cap.isOpened():
            ok,frm=cap.read()
            if not ok: break
            fh,fw=frm.shape[:2]
            if fw>1280:
                scale=1280/fw; frm=cv2.resize(frm,(int(fw*scale),int(fh*scale)))
            ann,_=dispatch(frm,task,selected_ppe=sel,reset=first,session_id=sid); first=False
            _,buf=cv2.imencode(".jpg",ann,[cv2.IMWRITE_JPEG_QUALITY,75])
            yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n"+buf.tobytes()+b"\r\n"
        cap.release()
    return Response(stream_with_context(gen()),mimetype="multipart/x-mixed-replace; boundary=f")

@app.route("/api/webcam/start",methods=["POST"])
def webcam_start():
    global _cam,_cam_task,_cam_ppe,_prev_gray,_tamper_alert_hold,_current_session_id
    data=request.get_json(silent=True) or {}
    _cam_task=data.get("task","ppe"); _cam_ppe=data.get("selected_ppe",["all"])
    _current_session_id=log_session(_cam_task,"webcam")

    if _cam_task=="tamper":
        _prev_gray=None; _tamper_alert_hold=0

    with _cam_lock:
        if _cam is None or not _cam.isOpened():

            # 🔥 MOBILE CAMERA CONNECTION
            _cam = cv2.VideoCapture("http://192.0.0.4:8080/video")

            _cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
            _cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
            _cam.set(cv2.CAP_PROP_FPS,30)
            _cam.set(cv2.CAP_PROP_BUFFERSIZE,1)

        if not _cam.isOpened():
            return jsonify(error="Mobile camera not connected"),500

    return jsonify(ok=True)

@app.route("/api/webcam/stop",methods=["POST"])
def webcam_stop():
    global _cam
    with _cam_lock:
        if _cam and _cam.isOpened(): _cam.release(); _cam=None
    return jsonify(ok=True)

@app.route("/api/webcam/task",methods=["POST"])
def webcam_task_route():
    global _cam_task,_cam_ppe,_current_session_id
    data=request.get_json(silent=True) or {}
    _cam_task=data.get("task",_cam_task); _cam_ppe=data.get("selected_ppe",_cam_ppe)
    _current_session_id=log_session(_cam_task,"webcam")
    return jsonify(task=_cam_task)

@app.route("/api/webcam/stream")
def webcam_stream():
    def gen():
        global _cam,_cam_task,_cam_ppe,_current_session_id; first=True
        while True:
            with _cam_lock:
                if _cam is None or not _cam.isOpened(): break
                ok,frm=_cam.read()
            if not ok: break
            do_reset=first and _cam_task!="tamper"
            ann,_=dispatch(frm,_cam_task,selected_ppe=_cam_ppe,reset=do_reset,session_id=_current_session_id)
            first=False
            _,buf=cv2.imencode(".jpg",ann,[cv2.IMWRITE_JPEG_QUALITY,78])
            yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n"+buf.tobytes()+b"\r\n"
    return Response(stream_with_context(gen()),mimetype="multipart/x-mixed-replace; boundary=f")

# ══════════════════════════════════════════════════════════════
#  ANALYTICS API ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/api/analytics/summary")
def analytics_summary():
    conn=db_conn()
    total_sessions=conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    total_persons =conn.execute("SELECT COUNT(*) FROM person_detections").fetchone()[0]
    total_ppe     =conn.execute("SELECT COUNT(*) FROM ppe_detections").fetchone()[0]
    violations    =conn.execute("SELECT COUNT(*) FROM ppe_detections WHERE is_violation=1").fetchone()[0]
    breaches      =conn.execute("SELECT COUNT(*) FROM tripwire_events WHERE breach=1").fetchone()[0]
    tamper_alerts =conn.execute("SELECT COUNT(*) FROM tamper_events WHERE tampered=1").fetchone()[0]
    conn.close()
    return jsonify(
        total_sessions=total_sessions,
        total_persons=total_persons,
        total_ppe=total_ppe,
        violations=violations,
        compliance_rate=round((1-violations/total_ppe)*100,1) if total_ppe>0 else 100,
        breaches=breaches,
        tamper_alerts=tamper_alerts
    )

@app.route("/api/analytics/ppe_class_counts")
def ppe_class_counts():
    task=request.args.get("task","all")
    conn=db_conn()
    if task=="all":
        rows=conn.execute("""
            SELECT class_name, COUNT(*) as cnt, SUM(is_violation) as violations
            FROM ppe_detections GROUP BY class_name ORDER BY cnt DESC
        """).fetchall()
    else:
        rows=conn.execute("""
            SELECT p.class_name, COUNT(*) as cnt, SUM(p.is_violation) as violations
            FROM ppe_detections p
            JOIN sessions s ON p.session_id=s.id
            WHERE s.task=?
            GROUP BY p.class_name ORDER BY cnt DESC
        """,(task,)).fetchall()
    conn.close()
    return jsonify([{"class":r["class_name"],"count":r["cnt"],"violations":r["violations"] or 0} for r in rows])

@app.route("/api/analytics/detections_over_time")
def detections_over_time():
    task=request.args.get("task","ppe")
    conn=db_conn()
    if task=="person":
        rows=conn.execute("""
            SELECT strftime('%Y-%m-%d %H', timestamp) as hr, COUNT(*) as cnt
            FROM person_detections GROUP BY hr ORDER BY hr
        """).fetchall()
    elif task=="tripwire":
        rows=conn.execute("""
            SELECT strftime('%Y-%m-%d %H', timestamp) as hr, SUM(breach) as cnt
            FROM tripwire_events GROUP BY hr ORDER BY hr
        """).fetchall()
    elif task=="tamper":
        rows=conn.execute("""
            SELECT strftime('%Y-%m-%d %H', timestamp) as hr, SUM(tampered) as cnt
            FROM tamper_events GROUP BY hr ORDER BY hr
        """).fetchall()
    else:
        rows=conn.execute("""
            SELECT strftime('%Y-%m-%d %H', timestamp) as hr,
                   COUNT(*) as cnt, SUM(is_violation) as violations
            FROM ppe_detections GROUP BY hr ORDER BY hr
        """).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/analytics/violations_vs_ok")
def violations_vs_ok():
    conn=db_conn()
    rows=conn.execute("""
        SELECT class_name,
               SUM(CASE WHEN is_violation=0 THEN 1 ELSE 0 END) as ok,
               SUM(is_violation) as violation
        FROM ppe_detections GROUP BY class_name
    """).fetchall()
    conn.close()
    return jsonify([{"class":r["class_name"],"ok":r["ok"],"violation":r["violation"]} for r in rows])

@app.route("/api/analytics/sessions_by_task")
def sessions_by_task():
    conn=db_conn()
    rows=conn.execute("SELECT task, COUNT(*) as cnt FROM sessions GROUP BY task").fetchall()
    conn.close()
    return jsonify([{"task":r["task"],"count":r["cnt"]} for r in rows])

@app.route("/api/analytics/breach_timeline")
def breach_timeline():
    conn=db_conn()
    rows=conn.execute("""
        SELECT strftime('%Y-%m-%d', timestamp) as day, SUM(breach) as breaches
        FROM tripwire_events GROUP BY day ORDER BY day
    """).fetchall()
    conn.close()
    return jsonify([{"day":r["day"],"breaches":r["breaches"]} for r in rows])

@app.route("/api/analytics/tamper_reasons")
def tamper_reasons():
    conn=db_conn()
    rows=conn.execute("SELECT reasons FROM tamper_events WHERE tampered=1").fetchall()
    conn.close()
    counts={}
    for r in rows:
        try:
            reasons=json.loads(r["reasons"])
            for reason in reasons:
                counts[reason]=counts.get(reason,0)+1
        except: pass
    return jsonify([{"reason":k,"count":v} for k,v in counts.items()])

@app.route("/api/analytics/recent_sessions")
def recent_sessions():
    conn=db_conn()
    rows=conn.execute("""
        SELECT id,timestamp,task,source FROM sessions
        ORDER BY id DESC LIMIT 20
    """).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/analytics/compliance_trend")
def compliance_trend():
    conn=db_conn()
    rows=conn.execute("""
        SELECT strftime('%Y-%m-%d', p.timestamp) as day,
               COUNT(*) as total,
               SUM(p.is_violation) as violations
        FROM ppe_detections p
        GROUP BY day ORDER BY day
    """).fetchall()
    conn.close()
    result=[]
    for r in rows:
        total=r["total"]; viol=r["violations"] or 0
        rate=round((1-viol/total)*100,1) if total>0 else 100
        result.append({"day":r["day"],"compliance_rate":rate,"total":total,"violations":viol})
    return jsonify(result)

if __name__=="__main__":
    init_db()
    load_model()
    print("[Warmup] Pre-warming models...")
    _dummy=np.zeros((480,640,3),dtype=np.uint8)
    _model(_dummy,verbose=False); _person_model(_dummy,verbose=False)
    print("[Warmup] Done!")
    app.run(debug=False,host="0.0.0.0",port=5000,threaded=True)