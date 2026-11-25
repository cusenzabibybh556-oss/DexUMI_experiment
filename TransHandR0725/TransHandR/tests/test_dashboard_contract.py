from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "scripts" / "dashboard"


def test_robot_preparation_describes_left_only_and_unvalidated_o6_start():
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    assert "只准备左臂" in html
    assert "不自动归位" in html
    assert "已在正式起点" in html
    assert "固定使用 ServoP" in html
    assert "不发送头部目标" in html
    assert "右腕不做 MoveP 工作空间验收" in html
    assert "沿基座 Z 抬高 8 cm" in html
    assert "新高度尚未实测" in html
    assert "ServoJ 因真机目标限幅异常已停用" in html
    assert "左臂 IK + ServoJ" not in html
    assert "双臂 MoveJ 到保存起点" not in html


def test_source_specific_visuals_and_single_channel_selector():
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    javascript = (DASHBOARD / "dashboard.js").read_text(encoding="utf-8")

    assert "LINKERFORCE · RAW JOINTS" not in html
    assert "operatorJoints" not in html
    assert "source-toggles" not in html
    assert html.count('id="channels"') == 1
    assert "25-POINT HAND" not in html
    assert 'id="inspireModelCard"' in html
    assert 'id="inspireHandModel"' in html
    assert 'id="o6TactileCard"' in html
    assert 'id="o6ModelCard"' in html
    assert "inspireModelCard').classList.toggle('hidden',!showInspire)" in javascript
    assert "showInspire=model==='inspire'" in javascript
    assert "showO6=source==='linkerforce'&&model==='o6'" in javascript
    assert "o6ModelCard').classList.toggle('hidden',!showO6)" in javascript
    assert 'id="deviceLeftGloveSerial"' in html
    assert 'id="deviceRightGloveSerial"' in html
    assert 'id="calibrationModal"' in html
    assert 'id="enableForceFeedback"' in html
    assert "showFeedback=source==='linkerforce'&&model==='inspire'" in javascript


def test_channel_draft_is_not_rebuilt_during_status_polling():
    javascript = (DASHBOARD / "dashboard.js").read_text(encoding="utf-8")
    system_poll = javascript.split("async function loadSystemConfig(){", 1)[1].split(
        "function markSystemDirty()", 1)[0]
    assert "updateChannelLock()" in system_poll
    assert "renderChannels()" not in system_poll
    live_poll = javascript.split("async function liveTick(){", 1)[1].split(
        "function renderValidity()", 1)[0]
    assert "updateChannelLock()" in live_poll
    assert "renderChannels()" not in live_poll


def test_inspire_viewer_uses_retargeted_six_channel_state_and_real_side_models():
    viewer = (DASHBOARD / "inspire_hand_viewer.js").read_text(encoding="utf-8")
    server = (ROOT / "scripts" / "realtime_multisource_dashboard.py").read_text(encoding="utf-8")
    assert "/assets/inspire/left/inspire_hand_left.urdf" in viewer
    assert "/assets/inspire/right/inspire_hand_right.urdf" in viewer
    assert "new GLTFLoader" in viewer
    assert "new STLLoader" not in viewer
    assert "MeshStandardMaterial" not in viewer
    assert '".glb": "model/gltf-binary"' in server
    assert '"/assets/vendor/GLTFLoader.js"' in server
    assert '"/assets/vendor/BufferGeometryUtils.js"' in server
    assert "pinky_proximal_joint" in viewer
    assert "thumb_proximal_yaw_joint" in viewer
    assert "data?.hand_state" in viewer
    assert "data?.action" in viewer
    assert "new THREE.OrthographicCamera" in viewer
    assert "new THREE.Color('#7b8282')" in viewer
    assert "center.x - distance" in viewer
    assert "links.get('hand_base_link')" in viewer
    assert "requestAnimationFrame(this.animate)" in viewer


def test_o6_viewer_uses_state_mapping_and_fingertip_tactile_patches():
    viewer = (DASHBOARD / "o6_hand_viewer.js").read_text(encoding="utf-8")
    assert "'thumb_cmc_pitch', 'thumb_cmc_yaw', 'index_mcp_pitch'" in viewer
    for link in ("thumb_distal", "index_distal", "middle_distal", "ring_distal", "pinky_distal"):
        assert link in viewer
    assert "data?.tactile?.matrix" in viewer
    assert "data?.tactile?.max" not in viewer
    assert "new THREE.DataTexture" in viewer
    assert "createPatchGeometry" in viewer
    assert "new THREE.OrthographicCamera" in viewer
    assert "now - this.lastTactileAt <= TACTILE_STALE_MS" in viewer
    assert "const PATCH_NEUTRAL = new THREE.Color('#b4bdb9')" in viewer
    assert "[0, neutral]" in viewer
    assert "data?.palm?.matrix" in viewer
    assert "PALM_ROWS = 14" in viewer and "PALM_COLS = 15" in viewer
    assert "palm_tactile_patch" in viewer
    assert "PALM_LAYOUT = { link: 'hand_base_link', flipRows: true, flipCols: false }" in viewer
    assert "const PALM_NEUTRAL = NEUTRAL.clone()" in viewer
    assert "createPalmPatchGeometry" in viewer
    assert "PALM_FOOTPRINT" in viewer
    assert "PALM_SENSOR_OUTLINE" not in viewer
    assert "palmFootprintEdges(height)" in viewer
    assert "PALM_MESH_ROWS = 42" in viewer and "PALM_MESH_COLS = 45" in viewer
    assert "createPalmFaceBins(sourceGeometry, bounds)" in viewer
    assert "palmPreview=zero" not in viewer
    assert "['zero', 'gradient', 'reference'].includes(PALM_PREVIEW)" in viewer
    assert "candidateFaces(y, z)" in viewer
    assert "palmMesh.name = 'palm_tactile_patch'" in viewer
    assert "PALM_WHITE_BANDS" not in viewer
    assert "createPalmWhiteGeometry" not in viewer
    assert "palmPressureColor(value)" in viewer
    assert "tipProgress" in viewer
    assert viewer.count("flipRows: true") == 6
    assert viewer.count("requestAnimationFrame(this.animate)") == 1
