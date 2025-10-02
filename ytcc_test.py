# 📊 유튜브 반응 리포트: AI 댓글요약 (Streamlit Cloud용 / 동시실행 1 슬롯 락 포함)

import streamlit as st
import pandas as pd
import io, os, json, re, time, shutil
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

import google.generativeai as genai

import plotly.express as px
from plotly import graph_objects as go
import circlify
import stopwordsiso as stopwords
from collections import Counter
from kiwipiepy import Kiwi
import numpy as np

try:
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
except Exception:
    ILLEGAL_CHARACTERS_RE = None

# ===================== 기본 경로(Cloud) =====================
BASE_DIR = "/tmp"  # Streamlit Cloud는 /tmp만 쓰기 가능(휘발성)
SESS_DIR = os.path.join(BASE_DIR, "sessions")
os.makedirs(SESS_DIR, exist_ok=True)

# ===================== 비밀키 / 파라미터 =====================
# secrets 우선, 없으면 하드코딩 백업
_YT_FALLBACK = []
_GEM_FALLBACK = []

YT_API_KEYS = list(st.secrets.get("YT_API_KEYS", [])) or _YT_FALLBACK
GEMINI_API_KEYS = list(st.secrets.get("GEMINI_API_KEYS", [])) or _GEM_FALLBACK
GEMINI_MODEL = "gemini-2.0-flash-lite"
GEMINI_TIMEOUT = 120
GEMINI_MAX_TOKENS = 2048

# Google Drive 설정
GDRIVE_PARENT_FOLDER_ID = st.secrets.get("GDRIVE_PARENT_FOLDER_ID", "")
_GDRIVE_KEYS_RAW = [st.secrets.get("GDRIVE_KEY_1"), st.secrets.get("GDRIVE_KEY_2"), st.secrets.get("GDRIVE_KEY_3")]
GDRIVE_KEYS = []
for raw in _GDRIVE_KEYS_RAW:
    if raw:
        try:
            GDRIVE_KEYS.append(json.loads(raw))
        except Exception:
            try:
                GDRIVE_KEYS.append(json.loads(str(raw)))
            except Exception:
                pass

# 수집 상한(필요시 조정)
MAX_TOTAL_COMMENTS = 200_000
MAX_COMMENTS_PER_VIDEO = 5_000

# ===================== 동시 실행 1 슬롯(락 파일) =====================
LOCK_PATH = os.path.join(BASE_DIR, "ytccai.busy.lock")

def try_acquire_lock(ttl=7200):
    # 오래된 락 정리
    if os.path.exists(LOCK_PATH):
        try:
            if time.time() - os.path.getmtime(LOCK_PATH) > ttl:
                os.remove(LOCK_PATH)
        except:
            pass
    if os.path.exists(LOCK_PATH):
        return False
    open(LOCK_PATH, "w").close()
    return True

def refresh_lock():
    try: os.utime(LOCK_PATH, None)
    except: pass

def release_lock():
    try:
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)
    except:
        pass

def lock_guard_start_or_warn():
    """긴 작업 시작 전에 호출: 락을 잡고 True 반환, 실패시 경고 후 False"""
    if not try_acquire_lock():
        st.warning("다른 사용자가 작업 중입니다. 잠시 후 다시 시도하세요.")
        return False
    return True

# ===================== 기본 UI =====================
st.set_page_config(page_title="📊 유튜브 반응 리포트: AI 댓글요약", layout="wide", initial_sidebar_state="collapsed")
st.title("📊 유튜브 반응 분석: AI 댓글요약")
st.caption("문의사항:미디어)디지털마케팅팀 데이터파트")

_YT_ID_RE = re.compile(r'^[A-Za-z0-9_-]{11}$')
def _kst_tz(): return timezone(timedelta(hours=9))
def kst_to_rfc3339_utc(dt_kst: datetime) -> str:
    if dt_kst.tzinfo is None: dt_kst = dt_kst.replace(tzinfo=_kst_tz())
    return dt_kst.astimezone(timezone.utc).isoformat().replace("+00:00","Z")

def clean_illegal(val):
    if isinstance(val, str) and ILLEGAL_CHARACTERS_RE is not None:
        return ILLEGAL_CHARACTERS_RE.sub('', val)
    return val

# ===================== 형태소/불용어 =====================
kiwi = Kiwi()
korean_stopwords = stopwords.stopwords("ko")

# ===================== (로그 제거) append_log → no-op =====================
def append_log(*args, **kwargs):
    # 로그 비활성화: 아무것도 하지 않음
    return

# ===================== 키 로테이터 (범용화) =====================
class RotatingKeys:
    def __init__(self, keys, state_key: str, on_rotate=None, treat_as_strings: bool = True):
        cleaned = []
        for k in (keys or []):
            if k is None:
                continue
            if treat_as_strings and isinstance(k, str):
                ks = k.strip()
                if ks:
                    cleaned.append(ks)
            else:
                cleaned.append(k)
        self.keys = cleaned[:10]
        self.state_key = state_key
        self.on_rotate = on_rotate
        idx = st.session_state.get(state_key, 0)
        self.idx = 0 if not self.keys else (idx % len(self.keys))
        st.session_state[state_key] = self.idx
    def current(self):
        if not self.keys: return None
        return self.keys[self.idx % len(self.keys)]
    def rotate(self):
        if not self.keys: return
        self.idx = (self.idx + 1) % len(self.keys)
        st.session_state[self.state_key] = self.idx
        if callable(self.on_rotate): self.on_rotate(self.idx, self.current())

# ===================== API 호출 래퍼 =====================
def is_youtube_quota_error(e: HttpError) -> bool:
    try:
        data = json.loads(getattr(e, "content", b"{}").decode("utf-8", errors="ignore"))
        status = getattr(getattr(e, 'resp', None), 'status', None)
        if status in (403, 429):
            reasons = [(err.get("reason") or "").lower() for err in data.get("error", {}).get("errors", [])]
            msg = (data.get("error", {}).get("message", "") or "").lower()
            quota_flags = ("quotaexceeded", "dailylimitexceeded", "ratelimitexceeded")
            if any(r in quota_flags for r in reasons): return True
            if "rate" in msg and "limit" in msg: return True
            if "quota" in msg: return True
        return False
    except Exception:
        return False

def with_retry(fn, tries=2, backoff=1.4):
    for i in range(tries):
        try:
            return fn()
        except HttpError as e:
            status = getattr(getattr(e, 'resp', None), 'status', None)
            if status in (400, 401, 403) and not is_youtube_quota_error(e):
                raise
            if i == tries - 1: raise
            time.sleep((i + 1) * backoff)
        except Exception:
            if i == tries - 1: raise
            time.sleep((i + 1) * backoff)

class RotatingYouTube:
    def __init__(self, keys, state_key="yt_key_idx", log=None):
        self.rot = RotatingKeys(keys, state_key, on_rotate=lambda i, k: log and log(f"🔁 YouTube 키 전환 → #{i+1}"))
        self.log = log
        self.service = None
        self._build_service()
    def _build_service(self):
        key = self.rot.current()
        if not key:
            raise RuntimeError("YouTube API Key가 비어 있습니다.")
        self.service = build("youtube", "v3", developerKey=key)
    def _rotate_and_rebuild(self):
        self.rot.rotate(); self._build_service()
    def execute(self, request_factory, tries_per_key=2):
        attempts = 0
        max_attempts = len(self.rot.keys) if self.rot.keys else 1
        while attempts < max_attempts:
            try:
                req = request_factory(self.service)
                return with_retry(lambda: req.execute(), tries=tries_per_key, backoff=1.4)
            except HttpError as e:
                if is_youtube_quota_error(e) and len(self.rot.keys) > 1:
                    self._rotate_and_rebuild()
                    attempts += 1
                    continue
                raise

def is_gemini_quota_error(exc: Exception) -> bool:
    msg = (str(exc) or "").lower()
    return ("429" in msg) or ("too many requests" in msg) or ("rate limit" in msg) or ("resource exhausted" in msg) or ("quota" in msg)

def call_gemini_rotating(model_name: str, keys, system_instruction: str, user_payload: str,
                         timeout_s: int = GEMINI_TIMEOUT, max_tokens: int = GEMINI_MAX_TOKENS, on_rotate=None) -> str:
    rot = RotatingKeys(keys, state_key="gem_key_idx", on_rotate=lambda i, k: on_rotate and on_rotate(i, k))
    if not rot.current():
        raise RuntimeError("Gemini API Key가 비어 있습니다.")
    attempts = 0
    max_attempts = len(rot.keys) if rot.keys else 1
    while attempts < max_attempts:
        try:
            genai.configure(api_key=rot.current())
            model = genai.GenerativeModel(
                model_name,
                generation_config={"temperature": 0.2, "max_output_tokens": max_tokens, "top_p": 0.9}
            )
            resp = model.generate_content([system_instruction, user_payload],
                                          request_options={"timeout": timeout_s})
            out = getattr(resp, "text", None)
            if not out and hasattr(resp, "candidates") and resp.candidates:
                c = resp.candidates[0]
                if hasattr(c, "content") and getattr(c.content, "parts", None):
                    part = c.content.parts[0]
                    if hasattr(part, "text"):
                        out = part.text
            return out or ""
        except Exception as e:
            if is_gemini_quota_error(e) and len(rot.keys) > 1:
                rot.rotate(); attempts += 1; continue
            raise

# ===================== Google Drive (서비스계정 로테이션) =====================
_GOOGLE_SCOPES = ["https://www.googleapis.com/auth/drive"]
def _build_drive_service_from_creds_dict(creds_dict: dict):
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=_GOOGLE_SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)

class RotatingDrive:
    def __init__(self, creds_dicts: list[dict], state_key="drive_key_idx", log=None):
        self.rot = RotatingKeys(
            list(range(len(creds_dicts))),
            state_key,
            on_rotate=lambda i, _: log and log(f"🔁 Drive 키 전환 → #{i+1}"),
            treat_as_strings=False,  # << 중요: 정수 인덱스를 문자열 취급하지 않음
        )
        self.creds_dicts = creds_dicts or []
        if not self.creds_dicts:
            raise RuntimeError("Drive 서비스 계정 키가 비어 있습니다.")
        self.service = None
        self._build()
        self.log = log
    def _build(self):
        self.service = _build_drive_service_from_creds_dict(self.creds_dicts[self.rot.idx])
    def _rotate_and_rebuild(self):
        self.rot.rotate()
        self._build()
    def execute(self, fn, tries_per_key=2):
        attempts = 0
        max_attempts = len(self.creds_dicts) if self.creds_dicts else 1
        while attempts < max_attempts:
            try:
                return with_retry(lambda: fn(self.service), tries=tries_per_key, backoff=1.6)
            except Exception:
                if attempts < max_attempts - 1:
                    self._rotate_and_rebuild()
                    attempts += 1
                    continue
                raise

def drive_create_folder(rd: RotatingDrive, name: str, parent_id: str) -> str:
    def _create(svc):
        meta = {"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
        return svc.files().create(body=meta, fields="id").execute()["id"]
    return rd.execute(_create)

def drive_upload_file(rd: RotatingDrive, folder_id: str, local_path: str, mime_type: str | None = None) -> dict:
    filename = os.path.basename(local_path)
    def _upload(svc):
        meta = {"name": filename, "parents": [folder_id]}
        media = MediaFileUpload(local_path, mimetype=mime_type, resumable=False)
        return svc.files().create(body=meta, media_body=media, fields="id, name, mimeType, webViewLink, webContentLink, size, createdTime").execute()
    return rd.execute(_upload)

def drive_list_folders(rd: RotatingDrive, parent_id: str) -> list[dict]:
    items = []
    page_token = None
    query = f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    def _list(svc, token):
        return svc.files().list(q=query, spaces="drive", fields="nextPageToken, files(id, name, createdTime)", pageToken=token, orderBy="name").execute()
    while True:
        resp = rd.execute(lambda svc: _list(svc, page_token))
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items

def drive_list_files_in_folder(rd: RotatingDrive, folder_id: str) -> list[dict]:
    items, page_token = [], None
    query = f"'{folder_id}' in parents and trashed=false"
    def _list(svc, token):
        return svc.files().list(q=query, spaces="drive", fields="nextPageToken, files(id, name, mimeType, size, webViewLink, webContentLink, createdTime)", pageToken=token, orderBy="name").execute()
    while True:
        resp = rd.execute(lambda svc: _list(svc, page_token))
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items

# ===================== 유틸: ID/URL =====================
def extract_video_id_one(s: str):
    s = (s or "").strip()
    if not s: return None
    if _YT_ID_RE.match(s): return s
    try:
        u = urlparse(s)
    except Exception:
        return None
    q = parse_qs(u.query or "")
    if u.path == "/watch" and "v" in q:
        v = q.get("v", [""])[0]
        return v if _YT_ID_RE.match(v) else None
    if u.netloc.endswith("youtu.be"):
        v = u.path.strip("/")
        return v if _YT_ID_RE.match(v) else None
    if "youtube.com" in u.netloc and "/embed/" in u.path:
        v = u.path.split("/embed/")[-1].split("/")[0]
        return v if _YT_ID_RE.match(v) else None
    if "youtube.com" in u.netloc and u.path.startswith("/shorts/"):
        v = u.path.split("/shorts/")[-1].split("/")[0]
        return v if _YT_ID_RE.match(v) else None
    return None

def extract_video_ids_from_text(text: str):
    ids = []
    for line in (text or "").splitlines():
        vid = extract_video_id_one(line)
        if vid and vid not in ids:
            ids.append(vid)
    return ids

# ===================== 직렬화/샘플링 =====================
def sample_max_5000(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    if n <= 5000:
        return df.copy().reset_index(drop=True)
    try:
        return df.nlargest(5000, "likeCount").reset_index(drop=True)
    except Exception:
        return df.head(5000).reset_index(drop=True)

def serialize_comments_for_llm(df: pd.DataFrame, max_rows=1500, max_chars_per_comment=280, max_total_chars=450_000):
    if df is None or df.empty:
        return "", 0, 0
    try:
        df2 = df.nlargest(max_rows, "likeCount") if "likeCount" in df.columns else df.head(max_rows)
    except Exception:
        df2 = df.head(max_rows)
    lines, total = [], 0
    for _, r in df2.iterrows():
        is_reply = "R" if int(r.get("isReply", 0) or 0) == 1 else "T"
        author = str(r.get("author", "") or "").replace("\n", " ")
        likec = int(r.get("likeCount", 0) or 0)
        text = str(r.get("text", "") or "").replace("\n", " ")
        if len(text) > max_chars_per_comment:
            text = text[:max_chars_per_comment] + "…"
        line = f"[{is_reply}|♥{likec}] {author}: {text}"
        if total + len(line) + 1 > max_total_chars:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines), len(lines), total

# ===================== YouTube API 함수 =====================
def yt_search_videos(rt, keyword, max_results, order="relevance", published_after=None, published_before=None, log=None):
    video_ids, token = [], None
    while len(video_ids) < max_results:
        params = dict(q=keyword, part="id", type="video", order=order, maxResults=min(50, max_results - len(video_ids)))
        if published_after: params["publishedAfter"] = published_after
        if published_before: params["publishedBefore"] = published_before
        if token: params["pageToken"] = token
        resp = rt.execute(lambda s: s.search().list(**params))
        for it in resp.get("items", []):
            vid = it["id"]["videoId"]
            if vid not in video_ids: video_ids.append(vid)
        token = resp.get("nextPageToken")
        if not token: break
        if log: log(f"검색 진행: {len(video_ids)}개")
        time.sleep(0.35)
    return video_ids

def yt_video_statistics(rt, video_ids, log=None):
    rows = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        resp = rt.execute(lambda s: s.videos().list(part="statistics,snippet,contentDetails", id=",".join(batch)))
        for item in resp.get("items", []):
            stats = item.get("statistics", {})
            snip = item.get("snippet", {})
            cont = item.get("contentDetails", {})
            dur_iso = cont.get("duration", "")
            def _dsec(dur: str):
                if not dur or not dur.startswith("P"): return None
                h = re.search(r"(\d+)H", dur); m = re.search(r"(\d+)M", dur); s = re.search(r"(\d+)S", dur)
                return (int(h.group(1)) if h else 0) * 3600 + (int(m.group(1)) if m else 0) * 60 + (int(s.group(1)) if s else 0)
            dur_sec = _dsec(dur_iso)
            short_type = "Shorts" if (dur_sec is not None and dur_sec <= 60) else "Clip"
            vid_id = item.get("id")
            rows.append({
                "video_id": vid_id,
                "video_url": f"https://www.youtube.com/watch?v={vid_id}",
                "title": snip.get("title", ""),
                "channelTitle": snip.get("channelTitle", ""),
                "publishedAt": snip.get("publishedAt", ""),
                "duration": dur_iso,
                "shortType": short_type,
                "viewCount": int(stats.get("viewCount", 0) or 0),
                "likeCount": int(stats.get("likeCount", 0) or 0),
                "commentCount": int(stats.get("commentCount", 0) or 0),
            })
        if log: log(f"통계 배치 {i // 50 + 1} 완료")
        time.sleep(0.35)
    return rows

def yt_all_replies(rt, parent_id, video_id, title="", short_type="Clip", log=None, cap=None):
    replies, token = [], None
    while True:
        if cap is not None and len(replies) >= cap:
            return replies[:cap]
        params = dict(part="snippet", parentId=parent_id, maxResults=100, pageToken=token, textFormat="plainText")
        try:
            resp = rt.execute(lambda s: s.comments().list(**params))
        except HttpError as e:
            if log: log(f"[오류] replies {video_id}/{parent_id}: {e}")
            break
        for c in resp.get("items", []):
            sn = c["snippet"]
            replies.append({
                "video_id": video_id, "video_title": title, "shortType": short_type,
                "comment_id": c.get("id", ""), "parent_id": parent_id, "isReply": 1,
                "author": sn.get("authorDisplayName", ""),
                "text": sn.get("textDisplay", "") or "",
                "publishedAt": sn.get("publishedAt", ""),
                "likeCount": int(sn.get("likeCount", 0) or 0),
            })
            if cap is not None and len(replies) >= cap:
                return replies[:cap]
        token = resp.get("nextPageToken")
        if not token: break
        time.sleep(0.25)
    return replies

def yt_all_comments_sync(rt, video_id, title="", short_type="Clip", include_replies=True, log=None,
                    max_per_video: int | None = None):
    rows, token = [], None
    while True:
        if max_per_video is not None and len(rows) >= max_per_video:
            return rows[:max_per_video]
        params = dict(part="snippet,replies", videoId=video_id, maxResults=100, pageToken=token, textFormat="plainText")
        try:
            resp = rt.execute(lambda s: s.commentThreads().list(**params))
        except HttpError as e:
            if log: log(f"[오류] commentThreads {video_id}: {e}")
            break

        for it in resp.get("items", []):
            top = it["snippet"]["topLevelComment"]["snippet"]
            thread_id = it["snippet"]["topLevelComment"]["id"]
            total_replies = int(it["snippet"].get("totalReplyCount", 0) or 0)
            rows.append({
                "video_id": video_id, "video_title": title, "shortType": short_type,
                "comment_id": thread_id, "parent_id": "", "isReply": 0,
                "author": top.get("authorDisplayName", ""),
                "text": top.get("textDisplay", "") or "",
                "publishedAt": top.get("publishedAt", ""),
                "likeCount": int(top.get("likeCount", 0) or 0),
            })
            if include_replies and total_replies > 0:
                cap = None
                if max_per_video is not None:
                    cap = max(0, max_per_video - len(rows))
                if cap == 0:
                    return rows[:max_per_video]
                rows.extend(yt_all_replies(rt, thread_id, video_id, title, short_type, log, cap=cap))
                if max_per_video is not None and len(rows) >= max_per_video:
                    return rows[:max_per_video]

        token = resp.get("nextPageToken")
        if not token: break
        if log: log(f"  댓글 페이지 진행, 누계 {len(rows)}")
        time.sleep(0.25)
    return rows

def parallel_collect_comments(video_list, rt_keys, include_replies, max_total_comments, max_per_video, log_callback, prog_callback):
    all_comments = []
    total_videos = len(video_list)
    collected_videos = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(
                yt_all_comments_sync,
                RotatingYouTube(rt_keys),
                vid_info["video_id"],
                vid_info.get("title", ""),
                vid_info.get("shortType", "Clip"),
                include_replies,
                None,
                max_per_video
            ): vid_info for vid_info in video_list
        }
        for future in as_completed(futures):
            vid_info = futures[future]
            try:
                comments = future.result()
                all_comments.extend(comments)
                collected_videos += 1
                if log_callback: log_callback(f"✅ [{collected_videos}/{total_videos}] {vid_info.get('title','')} - {len(comments):,}개 수집")
                if prog_callback: prog_callback(collected_videos / total_videos)
            except Exception as e:
                if log_callback: log_callback(f"❌ [{collected_videos+1}/{total_videos}] {vid_info.get('title','')} - 실패: {e}")
                collected_videos += 1
                if prog_callback: prog_callback(collected_videos / total_videos)
            if len(all_comments) >= max_total_comments:
                if log_callback: log_callback(f"최대 수집 한도({max_total_comments:,}개) 도달, 중단")
                break
    return all_comments[:max_total_comments]

# ===================== 세션 상태 =====================
def ensure_state():
    defaults = dict(
        focus_step=1,
        last_keyword="",
        # 심플
        s_query="", s_df_comments=None, s_df_analysis=None, s_df_stats=None,
        s_serialized_sample="", s_result_text="",
        s_history=[], s_preset="최근 1년",
        # 고급
        mode="검색 모드",
        df_stats=None, selected_ids=[],
        df_comments=None, df_analysis=None,
        adv_serialized_sample="", adv_result_text="",
        adv_followups=[], adv_history=[],
        # 입력값
        simple_follow_q="", adv_follow_q="",
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
ensure_state()

# ===================== 히스토리 → 컨텍스트 =====================
def build_history_context(pairs: list[tuple[str, str]]) -> str:
    if not pairs:
        return ""
    lines = []
    for i, (q, a) in enumerate(pairs, 1):
        lines.append(f"[이전 Q{i}]: {q}")   # ← 여기 닫는 괄호를 } 로
        lines.append(f"[이전 A{i}]: {a}")   # ← 이 줄도 동일 패턴
    return "\n".join(lines)


# ===================== 시각화 도구(저장용) =====================
def _fig_keyword_bubble(df_comments) -> go.Figure | None:
    try:
        custom_stopwords = {
            "아","휴","아이구","아이쿠","아이고","어","나","우리","저희","따라","의해","을","를",
            "에","의","가","으로","로","에게","뿐이다","의거하여","근거하여","입각하여","기준으로",
            "그냥","댓글","영상","오늘","이제","뭐","진짜","정말","부분","요즘","제발","완전",
            "그게","일단","모든","위해","대한","있지","이유","계속","실제","유튜브","이번","가장","드라마",
        }
        stopset = set(korean_stopwords); stopset.update(custom_stopwords)
        query_kw = (st.session_state.get("s_query")
                    or st.session_state.get("last_keyword")
                    or st.session_state.get("adv_analysis_keyword")
                    or "").strip()
        if query_kw:
            tokens_q = kiwi.tokenize(query_kw, normalize_coda=True)
            query_words = [t.form for t in tokens_q if t.tag in ("NNG","NNP") and len(t.form) > 1]
            stopset.update(query_words)

        texts = " ".join(df_comments["text"].astype(str).tolist())
        tokens = kiwi.tokenize(texts, normalize_coda=True)
        words = [t.form for t in tokens if t.tag in ("NNG","NNP") and len(t.form) > 1 and t.form not in stopset]
        counter = Counter(words)
        if len(counter) == 0:
            return None
        df_kw = pd.DataFrame(counter.most_common(30), columns=["word", "count"])
        df_kw["label"] = df_kw["word"] + "<br>" + df_kw["count"].astype(str)
        df_kw["scaled"] = np.sqrt(df_kw["count"])
        data_for_pack = [{"id": w, "datum": s} for w, s in zip(df_kw["word"], df_kw["scaled"])]
        circles = circlify.circlify(data_for_pack, show_enclosure=False,
                                    target_enclosure=circlify.Circle(x=0, y=0, r=1))
        pos = {c.ex["id"]: (c.x, c.y, c.r) for c in circles if "id" in c.ex}
        df_kw["x"] = df_kw["word"].map(lambda w: pos[w][0])
        df_kw["y"] = df_kw["word"].map(lambda w: pos[w][1])
        df_kw["r"] = df_kw["word"].map(lambda w: pos[w][2])
        min_size, max_size = 10, 22
        s_min, s_max = df_kw["scaled"].min(), df_kw["scaled"].max()
        df_kw["font_size"] = df_kw["scaled"].apply(
            lambda s: int(min_size + (s - s_min) / (s_max - s_min) * (max_size - min_size)) if s_max > s_min else 14
        )
        fig_kw = go.Figure()
        palette = px.colors.sequential.Blues
        df_kw["color_idx"] = df_kw["scaled"].apply(lambda s: int((s - s_min) / max(s_max - s_min, 1) * (len(palette) - 1)))
        for _, row in df_kw.iterrows():
            color = palette[int(row["color_idx"])]
            fig_kw.add_shape(
                type="circle", xref="x", yref="y",
                x0=row["x"] - row["r"], y0=row["y"] - row["r"],
                x1=row["x"] + row["r"], y1=row["y"] + row["r"],
                line=dict(width=0), fillcolor=color, opacity=0.88, layer="below"
            )
        fig_kw.add_trace(go.Scatter(
            x=df_kw["x"], y=df_kw["y"], mode="text",
            text=df_kw["label"], textposition="middle center",
            textfont=dict(color="white", size=df_kw["font_size"].tolist()),
            hovertext=df_kw["word"] + " (" + df_kw["count"].astype(str) + ")",
            hovertemplate="%{hovertext}<extra></extra>",
        ))
        fig_kw.update_xaxes(visible=False, range=[-1.05, 1.05])
        fig_kw.update_yaxes(visible=False, range=[-1.05, 1.05], scaleanchor="x", scaleratio=1)
        fig_kw.update_layout(title="Top30 키워드 버블", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=40, b=0))
        return fig_kw
    except Exception:
        return None

def _fig_time_series(df_comments, scope_label="(KST 기준)"):
    df_time = df_comments.copy()
    df_time["publishedAt"] = pd.to_datetime(df_time["publishedAt"], errors="coerce", utc=True).dt.tz_convert("Asia/Seoul")
    df_time = df_time.dropna(subset=["publishedAt"])
    if df_time.empty:
        return None
    span_hours = (df_time["publishedAt"].max() - df_time["publishedAt"].min()).total_seconds()/3600.0
    rule = "H" if span_hours <= 48 else "D"
    label = "시간별" if rule == "H" else "일자별"
    ts = df_time.resample(rule, on="publishedAt").size().reset_index(name="count")
    fig_ts = px.line(ts, x="publishedAt", y="count", markers=True, title=f"{label} 댓글량 추이 {scope_label}")
    return fig_ts

def _fig_top_videos(df_stats):
    if df_stats is None or df_stats.empty:
        return None
    top_vids = df_stats.sort_values(by="commentCount", ascending=False).head(10).copy()
    top_vids["title_short"] = top_vids["title"].apply(lambda t: t[:20] + "…" if isinstance(t, str) and len(t) > 20 else t)
    fig_vids = px.bar(top_vids, x="commentCount", y="title_short", orientation="h", text="commentCount",
                      title="Top10 영상 댓글수")
    return fig_vids

def _fig_top_authors(df_comments):
    if df_comments is None or df_comments.empty:
        return None
    top_authors = (df_comments.groupby("author").size().reset_index(name="count").sort_values(by="count", ascending=False).head(10))
    if top_authors.empty:
        return None
    fig_auth = px.bar(top_authors, x="count", y="author", orientation="h", text="count", title="Top10 댓글 작성자 활동량")
    return fig_auth

def _save_fig_png(fig: go.Figure, path: str):
    if fig is None:
        return False
    try:
        fig.write_image(path, format="png", scale=2)
        return True
    except Exception:
        return False

# ===================== 세션 저장/ZIP (로컬 + Drive 업로드) =====================
def _save_df_csv(df: pd.DataFrame, path: str):
    if df is None or (hasattr(df, "empty") and df.empty): return
    df.to_csv(path, index=False, encoding="utf-8-sig")

def _slugify_filename(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\-]+", "", s)
    if not s:
        s = "no_kw"
    return s[:60]

def _build_session_name() -> str:
    # 이름 포맷: 검색어_yyyy-mm-dd-hh:mm_검색기간 (KST)
    kw = (st.session_state.get("s_query") or st.session_state.get("last_keyword") or "").strip() or "no_kw"
    preset = (st.session_state.get("s_preset") or "최근 1년").replace(" ", "")
    now_kst = datetime.now(_kst_tz()).strftime("%Y-%m-%d-%H:%M")
    return f"{_slugify_filename(kw)}_{now_kst}_{preset}"

def _write_ai_texts(outdir: str):
    # ai_simple.md / ai_advanced.md
    simple_txt = st.session_state.get("s_result_text", "")
    adv_txt = st.session_state.get("adv_result_text", "")
    if simple_txt:
        with open(os.path.join(outdir, "ai_simple.md"), "w", encoding="utf-8") as f:
            f.write(simple_txt)
    if adv_txt:
        with open(os.path.join(outdir, "ai_advanced.md"), "w", encoding="utf-8") as f:
            f.write(adv_txt)

def _write_viz_pngs(outdir: str):
    # 현재 세션 데이터에서 그림 재생성 → PNG 저장
    df_comments_s = st.session_state.get("s_df_comments")
    df_stats_s = st.session_state.get("s_df_stats")
    df_comments_a = st.session_state.get("df_comments")
    df_stats_a = st.session_state.get("df_stats")

    # 심플 기준(있으면 우선)
    dfc = df_comments_s if (df_comments_s is not None and not df_comments_s.empty) else st.session_state.get("df_comments")
    dfs = df_stats_s if (df_stats_s is not None and not df_stats_s.empty) else st.session_state.get("df_stats")

    figs = {
        "viz_keyword_bubble.png": _fig_keyword_bubble(dfc) if dfc is not None and not dfc.empty else None,
        "viz_time_series.png": _fig_time_series(dfc) if dfc is not None and not dfc.empty else None,
        "viz_top_videos.png": _fig_top_videos(dfs) if dfs is not None and not dfs.empty else None,
        "viz_top_authors.png": _fig_top_authors(dfc) if dfc is not None and not dfc.empty else None,
    }
    for fname, fig in figs.items():
        _save_fig_png(fig, os.path.join(outdir, fname))

def save_current_session(name_prefix: str | None = None):
    # 새 포맷으로 세션 이름 생성
    sess_name = _build_session_name()
    outdir = os.path.join(SESS_DIR, sess_name)
    os.makedirs(outdir, exist_ok=True)

    # 메타/QA 저장
    qa_data = {
        "simple_history": st.session_state.get("s_history", []),
        "adv_history": st.session_state.get("adv_history", []),
        "simple_query": st.session_state.get("s_query",""),
        "last_keyword": st.session_state.get("last_keyword",""),
        "preset": st.session_state.get("s_preset",""),
        "saved_at_kst": datetime.now(_kst_tz()).strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(outdir, "qa.json"), "w", encoding="utf-8") as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)

    # CSV 저장
    _save_df_csv(st.session_state.get("s_df_comments"), os.path.join(outdir, "simple_comments_full.csv"))
    _save_df_csv(st.session_state.get("s_df_analysis"), os.path.join(outdir, "simple_comments_sample.csv"))
    _save_df_csv(st.session_state.get("s_df_stats"), os.path.join(outdir, "simple_videos.csv"))
    _save_df_csv(st.session_state.get("df_comments"), os.path.join(outdir, "adv_comments_full.csv"))
    _save_df_csv(st.session_state.get("df_analysis"), os.path.join(outdir, "adv_comments_sample.csv"))
    _save_df_csv(st.session_state.get("df_stats"), os.path.join(outdir, "adv_videos.csv"))

    # AI 텍스트 + 시각화 PNG 저장
    _write_ai_texts(outdir)
    _write_viz_pngs(outdir)

    # Drive 업로드 (세션 폴더 생성 후 전체 업로드)
    drive_folder_id = None
    uploaded = []
    if GDRIVE_PARENT_FOLDER_ID and GDRIVE_KEYS:
        try:
            rd = RotatingDrive(GDRIVE_KEYS, log=lambda m: st.write(m))
            drive_folder_id = drive_create_folder(rd, sess_name, GDRIVE_PARENT_FOLDER_ID)
            # 업로드
            mimemap = {
                ".csv": "text/csv",
                ".json": "application/json",
                ".md": "text/markdown",
                ".png": "image/png",
                ".zip": "application/zip"
            }
            for fn in sorted(os.listdir(outdir)):
                p = os.path.join(outdir, fn)
                if not os.path.isfile(p):
                    continue
                ext = os.path.splitext(fn)[1].lower()
                mime = mimemap.get(ext, "application/octet-stream")
                info = drive_upload_file(rd, drive_folder_id, p, mime)
                uploaded.append(info)
            # manifest.json 업로드
            manifest = {
                "session_name": sess_name,
                "parent_folder_id": GDRIVE_PARENT_FOLDER_ID,
                "drive_folder_id": drive_folder_id,
                "uploaded": uploaded,
                "created_kst": datetime.now(_kst_tz()).strftime("%Y-%m-%d %H:%M:%S")
            }
            man_local = os.path.join(outdir, "manifest.json")
            with open(man_local, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            drive_upload_file(rd, drive_folder_id, man_local, "application/json")
        except Exception as e:
            st.warning(f"Drive 업로드 중 문제 발생: {e}")

    return sess_name, drive_folder_id

def list_sessions_local():
    if not os.path.exists(SESS_DIR): return []
    return sorted([d for d in os.listdir(SESS_DIR) if os.path.isdir(os.path.join(SESS_DIR,d))], reverse=True)

def zip_session(sess_name: str):
    sess_path = os.path.join(SESS_DIR, sess_name)
    zip_path = os.path.join(SESS_DIR, f"{sess_name}.zip")
    shutil.make_archive(zip_path.replace(".zip",""), 'zip', sess_path)
    return zip_path

# ===================== 시각화/다운로드(화면 렌더) =====================
def render_keyword_bubble(s_df_comments):
    st.subheader("① 키워드 버블")
    try:
        custom_stopwords = {
            "아","휴","아이구","아이쿠","아이고","어","나","우리","저희","따라","의해","을","를",
            "에","의","가","으로","로","에게","뿐이다","의거하여","근거하여","입각하여","기준으로",
            "그냥","댓글","영상","오늘","이제","뭐","진짜","정말","부분","요즘","제발","완전",
            "그게","일단","모든","위해","대한","있지","이유","계속","실제","유튜브","이번","가장","드라마",
        }
        stopset = set(korean_stopwords); stopset.update(custom_stopwords)

        # 🔑 검색어 불용어 추가
        query_kw = (st.session_state.get("s_query") 
                    or st.session_state.get("last_keyword") 
                    or st.session_state.get("adv_analysis_keyword") 
                    or "").strip()
        if query_kw:
            tokens_q = kiwi.tokenize(query_kw, normalize_coda=True)
            query_words = [t.form for t in tokens_q if t.tag in ("NNG","NNP") and len(t.form) > 1]
            stopset.update(query_words)

        texts = " ".join(s_df_comments["text"].astype(str).tolist())
        tokens = kiwi.tokenize(texts, normalize_coda=True)
        words = [t.form for t in tokens if t.tag in ("NNG","NNP") and len(t.form) > 1 and t.form not in stopset]
        counter = Counter(words)
        if len(counter) == 0:
            st.info("표시할 키워드가 없습니다(불용어 제거 후 남은 단어 없음).")
            return
        df_kw = pd.DataFrame(counter.most_common(30), columns=["word", "count"])
        df_kw["label"] = df_kw["word"] + "<br>" + df_kw["count"].astype(str)
        df_kw["scaled"] = np.sqrt(df_kw["count"])
        data_for_pack = [{"id": w, "datum": s} for w, s in zip(df_kw["word"], df_kw["scaled"])]
        circles = circlify.circlify(data_for_pack, show_enclosure=False,
                                    target_enclosure=circlify.Circle(x=0, y=0, r=1))
        pos = {c.ex["id"]: (c.x, c.y, c.r) for c in circles if "id" in c.ex}
        df_kw["x"] = df_kw["word"].map(lambda w: pos[w][0])
        df_kw["y"] = df_kw["word"].map(lambda w: pos[w][1])
        df_kw["r"] = df_kw["word"].map(lambda w: pos[w][2])
        min_size, max_size = 10, 22
        s_min, s_max = df_kw["scaled"].min(), df_kw["scaled"].max()
        df_kw["font_size"] = df_kw["scaled"].apply(
            lambda s: int(min_size + (s - s_min) / (s_max - s_min) * (max_size - min_size)) if s_max > s_min else 14
        )
        fig_kw = go.Figure()
        palette = px.colors.sequential.Blues
        df_kw["color_idx"] = df_kw["scaled"].apply(lambda s: int((s - s_min) / max(s_max - s_min, 1) * (len(palette) - 1)))
        for _, row in df_kw.iterrows():
            color = palette[int(row["color_idx"])]
            fig_kw.add_shape(
                type="circle", xref="x", yref="y",
                x0=row["x"] - row["r"], y0=row["y"] - row["r"],
                x1=row["x"] + row["r"], y1=row["y"] + row["r"],
                line=dict(width=0), fillcolor=color, opacity=0.88, layer="below"
            )
        fig_kw.add_trace(go.Scatter(
            x=df_kw["x"], y=df_kw["y"], mode="text",
            text=df_kw["label"], textposition="middle center",
            textfont=dict(color="white", size=df_kw["font_size"].tolist()),
            hovertext=df_kw["word"] + " (" + df_kw["count"].astype(str) + ")",
            hovertemplate="%{hovertext}<extra></extra>",
        ))
        fig_kw.update_xaxes(visible=False, range=[-1.05, 1.05])
        fig_kw.update_yaxes(visible=False, range=[-1.05, 1.05], scaleanchor="x", scaleratio=1)
        fig_kw.update_layout(title="Top30 키워드 버블", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_kw, use_container_width=True)
    except Exception as e:
        st.info(f"키워드 분석 불가: {e}")

def render_quant_viz(df_comments, df_stats, scope_label="(KST 기준)"):
    if df_comments is not None and not df_comments.empty:
        with st.expander("📊 정량 요약", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    render_keyword_bubble(df_comments)
            with col2:
                with st.container(border=True):
                    st.subheader("② 시점별 댓글량 변동 추이")
                    df_time = df_comments.copy()
                    df_time["publishedAt"] = (
                        pd.to_datetime(df_time["publishedAt"], errors="coerce", utc=True).dt.tz_convert("Asia/Seoul")
                    )
                    df_time = df_time.dropna(subset=["publishedAt"])
                    if not df_time.empty:
                        span_hours = (df_time["publishedAt"].max() - df_time["publishedAt"].min()).total_seconds()/3600.0
                        rule = "H" if span_hours <= 48 else "D"
                        label = "시간별" if rule == "H" else "일자별"
                        ts = df_time.resample(rule, on="publishedAt").size().reset_index(name="count")
                        fig_ts = px.line(ts, x="publishedAt", y="count", markers=True,
                                         title=f"{label} 댓글량 추이 {scope_label}")
                        st.plotly_chart(fig_ts, use_container_width=True)
                    else:
                        st.info("댓글 타임스탬프가 비어 있습니다.")
            if df_stats is not None and not df_stats.empty:
                col3, col4 = st.columns(2)
                with col3:
                    with st.container(border=True):
                        st.subheader("③ Top10 영상 댓글수")
                        top_vids = df_stats.sort_values(by="commentCount", ascending=False).head(10).copy()
                        top_vids["title_short"] = top_vids.apply(
                            lambda r: f'<a href="https://www.youtube.com/watch?v={r["video_id"]}" target="_blank" '
                                      f'style="color:black; text-decoration:none;">'
                                      f'{r["title"][:20] + "…" if len(r["title"]) > 20 else r["title"]}</a>',
                            axis=1
                        )
                        fig_vids = px.bar(top_vids, x="commentCount", y="title_short",
                                          orientation="h", text="commentCount",
                                          hover_data={"title": True, "video_id": False})
                        st.plotly_chart(fig_vids, use_container_width=True)
                with col4:
                    with st.container(border=True):
                        st.subheader("④ 댓글 작성자 활동량 Top10")
                        top_authors = (
                            df_comments.groupby("author").size().reset_index(name="count")
                            .sort_values(by="count", ascending=False).head(10)
                        )
                        fig_auth = px.bar(top_authors, x="count", y="author", orientation="h", text="count",
                                          title="Top10 댓글 작성자 활동량")
                        st.plotly_chart(fig_auth, use_container_width=True)
                        user_counts = df_comments.groupby("author").size()
                        total_users = len(user_counts)
                        active_users = user_counts[user_counts >= 3].count()
                        perc = (active_users / total_users * 100) if total_users > 0 else 0
                        st.markdown(f"**3회 이상 댓글 작성 유저 비중:** {active_users:,}명 / {total_users:,}명 ({perc:.1f}%)")
            with st.container(border=True):
                st.subheader("⑤ 댓글 좋아요 Top10")
                top_comments = df_comments.sort_values(by="likeCount", ascending=False).head(10)
                for _, row in top_comments.iterrows():
                    url = f"https://www.youtube.com/watch?v={row['video_id']}"
                    st.markdown(
                        f"<div style='margin-bottom:15px;'>"
                        f"<b>{row['likeCount']} 👍</b> — {row['author']}<br>"
                        f"<span style='font-size:14px;'>▶️ <a href='{url}' target='_blank' style='color:black; text-decoration:none;'>"
                        f"{row.get('video_title','(제목없음)')}</a></span><br>"
                        f"> {row['text'][:150]}{'…' if len(row['text'])>150 else ''}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

def render_downloads(df_comments, df_analysis, df_stats, prefix="simple"):
    if df_comments is not None and not df_comments.empty:
        st.markdown("---")
        st.subheader("⬇️ 결과 다운로드")
        csv_full = df_comments.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "전체 댓글 (CSV)", data=csv_full,
            file_name=f"{prefix}_full_{len(df_comments)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", key=f"{prefix}_dl_full_csv"
        )
        if df_analysis is not None:
            csv_sample = df_analysis.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "분석용 샘플 (CSV)", data=csv_sample,
                file_name=f"{prefix}_sample_{len(df_analysis)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv", key=f"{prefix}_dl_sample_csv"
            )
        if df_stats is not None and not df_stats.empty:
            df_videos = df_stats.copy()
            if "viewCount" in df_videos.columns:
                df_videos = df_videos.sort_values(by="viewCount", ascending=False).reset_index(drop=True)
            csv_videos = df_videos.applymap(clean_illegal).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "영상목록 (CSV)", data=csv_videos,
                file_name=f"{prefix}_videolist_{len(df_videos)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv", key=f"{prefix}_dl_videos_csv"
            )

# ===================== 탭 =====================
tab_simple, tab_advanced, tab_sessions = st.tabs(["🟢 심플 모드", "⚙️ 고급 모드", "📂 세션 아카이브"])

# ===================== 추가질문 핸들러 =====================
def handle_followup_simple():
    follow_q = (st.session_state.get("simple_follow_q") or "").strip()
    if not follow_q: return
    if not GEMINI_API_KEYS:
        st.error("Gemini API Key가 없습니다."); return
    if not st.session_state.get("s_serialized_sample"):
        st.error("분석 샘플이 없습니다. 먼저 수집/분석 실행."); return
    append_log("심플-추가", st.session_state.get("s_query",""), follow_q)  # no-op
    context_str = build_history_context(st.session_state.get("s_history", []))
    system_instruction = (
        "너는 유튜브 댓글을 분석하는 어시스턴트다. "
        "아래는 이미 추출/가공된 댓글 샘플과 이전 질의응답 히스토리다. "
        "이전 맥락을 모두 고려하여 한국어로 간결하고 구조화된 답을 하라. "
        "반드시 모든 댓글을 읽고 답변하라."
    )
    payload = ((context_str + "\n\n") if context_str else "") + (
        f"[현재 질문]: {follow_q}\n"
        f"[기간]: {st.session_state.get('s_preset','최근 1년')}\n\n"
        f"[댓글 샘플]:\n{st.session_state['s_serialized_sample']}\n"
    )
    out = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, system_instruction, payload)
    st.session_state["s_history"].append((follow_q, out))
    st.session_state["simple_follow_q"] = ""
    st.success("추가 분석 완료")

def handle_followup_advanced():
    adv_follow_q = (st.session_state.get("adv_follow_q") or "").strip()
    if not adv_follow_q: return
    if not GEMINI_API_KEYS:
        st.error("Gemini API Key가 없습니다."); return
    df_analysis = st.session_state.get("df_analysis")
    if df_analysis is None or df_analysis.empty:
        st.error("분석 샘플이 없습니다. 먼저 수집/분석 실행."); return
    append_log("고급-추가", st.session_state.get("last_keyword",""), adv_follow_q)  # no-op
    a_text = st.session_state.get("adv_serialized_sample", "") or serialize_comments_for_llm(df_analysis)[0]
    context_str = build_history_context(st.session_state.get("adv_history", []))
    system_instruction = (
        "너는 유튜브 댓글을 분석하는 어시스턴트다. "
        "아래는 이미 추출/가공된 댓글 샘플과 이전 질의응답 히스토리다. "
        "이전 맥락을 모두 고려하여 한국어로 간결하고 구조화된 답을 하라. "
        "반드시 모든 댓글을 읽고 답변하라."
    )
    payload = ((context_str + "\n\n") if context_str else "") + (
        f"[현재 질문]: {adv_follow_q}\n\n"
        f"[댓글 샘플]:\n{a_text}\n"
    )
    out = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, system_instruction, payload)
    st.session_state["adv_followups"].append((adv_follow_q, out))
    st.session_state["adv_history"].append((adv_follow_q, out))
    st.session_state["adv_follow_q"] = ""
    st.success("추가 분석 완료(고급)")

# ===================== 1) 심플 모드 =====================
with tab_simple:
    st.subheader("최근 기간 댓글 반응 — 드라마/배우명으로 바로 분석")
    s_query = st.text_input("드라마 or 배우명", value=st.session_state.get("s_query", ""),
                            placeholder="키워드 입력", key="simple_query")
    preset_simple = st.radio(
        "업로드 기간 (KST)",
        ["최근 12시간","최근 24시간","최근 48시간","최근 1주일","최근 1개월","최근 6개월",
         "최근 1년","최근 2년","최근 3년","최근 4년","최근 5년","최근 10년"],
        horizontal=True, key="simple_preset"
    )
    user_question = st.text_area("추가 질문/요청(선택, 비우면 기본 질문)", height=80,
                                 placeholder="예: 연기력/호불호 중심으로 분석해줘", key="simple_question")

    SIMPLE_TOP_N = 50
    SIMPLE_ORDER = "viewCount"
    now_kst = datetime.now(_kst_tz())

    if preset_simple == "최근 12시간":      start_dt = now_kst - timedelta(hours=12)
    elif preset_simple == "최근 24시간":     start_dt = now_kst - timedelta(hours=24)
    elif preset_simple == "최근 48시간":     start_dt = now_kst - timedelta(hours=48)
    elif preset_simple == "최근 1주일":     start_dt = now_kst - timedelta(days=7)
    elif preset_simple == "최근 1개월":     start_dt = now_kst - timedelta(days=30)
    elif preset_simple == "최근 6개월":     start_dt = now_kst - timedelta(days=182)
    elif preset_simple == "최근 1년":       start_dt = now_kst - timedelta(days=365)
    elif preset_simple == "최근 2년":       start_dt = now_kst - timedelta(days=365*2)
    elif preset_simple == "최근 3년":       start_dt = now_kst - timedelta(days=365*3)
    elif preset_simple == "최근 4년":       start_dt = now_kst - timedelta(days=365*4)
    elif preset_simple == "최근 5년":       start_dt = now_kst - timedelta(days=365*5)
    elif preset_simple == "최근 10년":      start_dt = now_kst - timedelta(days=365*10)
    else:                                   start_dt = now_kst - timedelta(days=365)

    published_after = kst_to_rfc3339_utc(start_dt)
    published_before = kst_to_rfc3339_utc(now_kst)

    if st.button("🚀 분석하기", type="primary", key="simple_run"):
        if not YT_API_KEYS:
            st.error("YouTube API Key가 없습니다.")
        elif not GEMINI_API_KEYS:
            st.error("Gemini API Key가 없습니다.")
        elif not st.session_state["simple_query"].strip():
            st.warning("드라마 or 배우명을 입력하세요.")
        else:
            # === 동시 실행 락 시도 ===
            if not lock_guard_start_or_warn():
                st.stop()
            try:
                st.session_state["s_query"] = st.session_state["simple_query"].strip()
                st.session_state["s_preset"] = preset_simple
                st.session_state["s_history"] = []
                append_log("심플", st.session_state["s_query"], st.session_state.get("simple_question", ""))  # no-op

                status_ph = st.empty()
                with status_ph.status("심플 모드 실행 중…", expanded=True) as status:
                    rt = RotatingYouTube(YT_API_KEYS, log=lambda m: status.write(m))
                    status.write(f"🔍 영상 검색 중… ({preset_simple}, 정렬: {SIMPLE_ORDER})")
                    ids = yt_search_videos(rt, st.session_state["s_query"], SIMPLE_TOP_N,
                                           SIMPLE_ORDER, published_after, published_before, log=status.write)

                    status.write(f"🎞️ 대상 영상: {len(ids)} — 메타 조회…")
                    stats = yt_video_statistics(rt, ids, log=status.write)
                    df_stats = pd.DataFrame(stats)
                    st.session_state["s_df_stats"] = df_stats

                    # 병렬 댓글 수집 (대댓글 제외)
                    status.write("💬 댓글 수집 중…")
                    video_list = df_stats.to_dict('records')
                    prog = st.progress(0, text="수집 진행 중")
                    log_ph = st.empty()
                    rows = parallel_collect_comments(
                        video_list=video_list,
                        rt_keys=YT_API_KEYS,
                        include_replies=False,
                        max_total_comments=MAX_TOTAL_COMMENTS,
                        max_per_video=MAX_COMMENTS_PER_VIDEO,
                        log_callback=log_ph.write,
                        prog_callback=prog.progress
                    )

                    if not rows:
                        status.update(label="⚠️ 댓글을 수집하지 못했습니다.", state="error")
                        st.session_state["s_df_comments"] = None
                        st.session_state["s_df_analysis"] = None
                        st.session_state["s_serialized_sample"] = ""
                        st.session_state["s_result_text"] = ""
                    else:
                        df_comments = pd.DataFrame(rows).applymap(clean_illegal)
                        st.session_state["s_df_comments"] = df_comments
                        df_analysis = sample_max_5000(df_comments)
                        st.session_state["s_df_analysis"] = df_analysis

                        s_text, _, _ = serialize_comments_for_llm(
                            df_analysis, max_rows=1500, max_chars_per_comment=280, max_total_chars=450_000
                        )
                        st.session_state["s_serialized_sample"] = s_text

                        # Gemini 분석
                        status.write("🧠 AI 분석 중…")
                        system_instruction = (
                            "너는 유튜브 댓글을 분석하는 어시스턴트다. "
                            "아래 키워드와 지정된 기간 내 댓글 샘플을 바탕으로, 전반적 반응을 한국어로 간결하게 요약하라. "
                            "핵심 포인트를 항목화하고, 긍/부정/중립의 대략적 비율과 대표 코멘트(10개미만)를 예시로 제시하라. "
                            "키워드가 인물명이면 인물 중심, 드라마명이면 드라마 중심으로 분석하라. "
                            "반드시 모든 댓글을 읽고 답변하라."
                        )
                        default_q = f"{preset_simple} 기준으로 '{st.session_state['s_query']}'에 대한 유튜브 댓글 반응을 요약해줘."
                        prompt_q = (st.session_state.get("simple_question", "").strip() or default_q)
                        payload = (
                            f"[키워드]: {st.session_state['s_query']}\n"
                            f"[질문]: {prompt_q}\n"
                            f"[기간]: {preset_simple}\n\n"
                            f"[댓글 샘플]:\n{s_text}\n"
                        )
                        out = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, system_instruction, payload,
                                                   timeout_s=GEMINI_TIMEOUT, max_tokens=GEMINI_MAX_TOKENS,
                                                   on_rotate=lambda i, k: status.write(f"🔁 Gemini 키 전환 → #{i+1}"))
                        st.session_state["s_result_text"] = out
                        st.session_state["s_history"].append((prompt_q, out))
                        status.update(label="🎉 분석 완료", state="complete")
                status_ph.empty()
            finally:
                release_lock()

    # 결과/추가질문/시각화/다운로드
    s_df_comments = st.session_state.get("s_df_comments")
    s_df_analysis = st.session_state.get("s_df_analysis")
    s_df_stats = st.session_state.get("s_df_stats")

    if s_df_comments is not None and not s_df_comments.empty:
        st.success(f"수집 완료 — 전체 {len(s_df_comments):,}개 / 샘플 {len(s_df_analysis):,}개")

    if st.session_state.get("s_result_text"):
        with st.expander("🧠 AI 분석 결과", expanded=True):
            st.markdown(st.session_state["s_result_text"])
        if st.session_state.get("s_history"):
            st.markdown("### 📝 추가 질문 히스토리")
            for i, (q, a) in enumerate(st.session_state["s_history"][1:], start=1):
                with st.expander(f"Q{i}. {q}", expanded=True):
                    st.markdown(a or "_응답 없음_")
        st.markdown("#### ➕ 추가 질문하기")
        st.text_input("추가 질문", placeholder="예: 주연배우들에 대한 반응은 어때?",
                      key="simple_follow_q", on_change=handle_followup_simple)
        st.button("질문 보내기", key="simple_follow_btn", on_click=handle_followup_simple)

    render_quant_viz(s_df_comments, s_df_stats, scope_label="(KST 기준)")
    render_downloads(s_df_comments, s_df_analysis, s_df_stats, prefix="simple")

    if st.button("💾 세션 저장하기", key="simple_save_session"):
        name, drive_id = save_current_session(None)
        if drive_id:
            st.success(f"세션 저장 완료 · Drive 폴더: https://drive.google.com/drive/folders/{drive_id}")
        else:
            st.success(f"세션 저장 완료(로컬) · {name}")

# ===================== 2) 고급 모드 =====================
with tab_advanced:
    st.subheader("고급 모드 — 4단계로 세밀 제어 (심플과 동등 로직/시각화)")

    mode = st.radio("모드", ["검색 모드", "URL 직접 입력 모드"],
                    index=(0 if st.session_state.get("mode", "검색 모드") == "검색 모드" else 1),
                    horizontal=True, key="adv_mode_radio")
    if mode != st.session_state["mode"]:
        st.session_state["mode"] = mode
        st.session_state["focus_step"] = 1

    include_replies = st.checkbox("대댓글 포함", value=False, key="adv_include_replies")

    # ① 영상목록추출
    expanded1 = (st.session_state["focus_step"] == 1)
    with st.expander("① 영상목록추출", expanded=expanded1):
        published_after = published_before = None
        if st.session_state["mode"] == "검색 모드":
            st.markdown("**업로드 기간 (KST)**")
            preset = st.radio("프리셋", ["최근 12시간", "최근 30일", "최근 1년", "직접 입력"],
                              horizontal=True, key="adv_preset")
            now_kst = datetime.now(_kst_tz())
            if preset == "최근 12시간":
                start_dt = now_kst - timedelta(hours=12); end_dt = now_kst
            elif preset == "최근 30일":
                start_dt = now_kst - timedelta(days=30); end_dt = now_kst
            elif preset == "최근 1년":
                start_dt = now_kst - timedelta(days=365); end_dt = now_kst
            else:
                c1, c2 = st.columns(2)
                sd = c1.date_input("시작일", now_kst.date()-timedelta(days=30), key="adv_sd")
                stime = c1.time_input("시작 시:분", value=datetime.min.time().replace(hour=0, minute=0), key="adv_stime")
                ed = c2.date_input("종료일", now_kst.date(), key="adv_ed")
                etime = c2.time_input("종료 시:분", value=datetime.min.time().replace(hour=23, minute=59), key="adv_etime")
                start_dt = datetime.combine(sd, stime, tzinfo=_kst_tz())
                end_dt = datetime.combine(ed, etime, tzinfo=_kst_tz())
            published_after = kst_to_rfc3339_utc(start_dt)
            published_before = kst_to_rfc3339_utc(end_dt)

            c1, c2, c3 = st.columns([3, 1, 1])
            keyword = c1.text_input("검색 키워드", st.session_state.get("last_keyword", "") or "", key="adv_keyword")
            top_n = c2.number_input("TOP N", min_value=1, value=50, step=1, key="adv_topn")
            order = c3.selectbox("정렬", ["relevance", "viewCount"], key="adv_order")
        else:
            keyword = None; top_n = None; order = None
            urls_main = st.text_area("URL/ID 목록 (줄바꿈 구분)", height=160,
                                     placeholder="https://youtu.be/XXXXXXXXXXX\nXXXXXXXXXXX\n...", key="adv_urls")

        if st.button("목록 가져오기", use_container_width=True, key="adv_fetch_list"):
            if not YT_API_KEYS:
                st.error("YouTube API Key가 없습니다.")
            else:
                rt = RotatingYouTube(YT_API_KEYS, log=lambda m: st.write(m))
                log_box = st.empty()
                def log(msg): log_box.write(msg)
                if st.session_state["mode"] == "검색 모드":
                    st.session_state["last_keyword"] = keyword or ""
                    log("🔍 검색 실행 중…")
                    ids = yt_search_videos(rt, keyword, int(top_n), order, published_after, published_before, log)
                else:
                    ids = extract_video_ids_from_text(urls_main or "")
                    if not ids:
                        st.warning("URL/ID가 비어 있습니다."); st.stop()
                log(f"🎞️ 대상 영상 수: {len(ids)} — 메타/통계 조회…")
                stats = yt_video_statistics(rt, ids, log)
                df = pd.DataFrame(stats)
                if not df.empty and "publishedAt" in df.columns:
                    df["publishedAt"] = (
                        pd.to_datetime(df["publishedAt"], errors="coerce", utc=True)
                        .dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d %H:%M:%S")
                    )
                if "viewCount" in df.columns:
                    df = df.sort_values(by="viewCount", ascending=False).reset_index(drop=True)
                st.session_state["df_stats"] = df
                st.session_state["selected_ids"] = df["video_id"].tolist() if not df.empty else []
                st.session_state["focus_step"] = 2
                st.success(f"목록 준비 완료 — 총 {len(df)}개")
                st.rerun()

    # ② 영상선택 및 URL추가
    expanded2 = (st.session_state["focus_step"] == 2)
    with st.expander("② 영상선택 및 URL추가", expanded=expanded2):
        df_stats = st.session_state["df_stats"]
        if df_stats is None or df_stats.empty:
            st.info("①에서 먼저 목록을 가져오세요.")
        else:
            df_show = df_stats.copy()
            df_show["select"] = df_show["video_id"].apply(lambda v: v in st.session_state["selected_ids"])
            df_view = df_show[["video_id","select","title","channelTitle","shortType","viewCount","commentCount","publishedAt","video_url"]].set_index("video_id")
            edited = st.data_editor(
                df_view,
                column_config={
                    "select": st.column_config.CheckboxColumn("선택", default=True),
                    "title": st.column_config.TextColumn("제목"),
                    "channelTitle": st.column_config.TextColumn("채널"),
                    "shortType": st.column_config.TextColumn("타입"),
                    "viewCount": st.column_config.NumberColumn("조회수", format="%,d"),
                    "commentCount": st.column_config.NumberColumn("댓글수", format="%,d"),
                    "publishedAt": st.column_config.TextColumn("업로드(KST)"),
                    "video_url": st.column_config.LinkColumn("유튜브 링크", display_text="보기"),
                },
                use_container_width=True, num_rows="fixed", hide_index=False, key="adv_editor_table",
            )
            st.session_state["selected_ids"] = [vid for vid, row in edited.iterrows() if bool(row.get("select", False))]
            st.caption(f"선택된 영상: {len(st.session_state['selected_ids'])} / {len(edited)}")
            csel1, csel2, _ = st.columns([1,1,6])
            if csel1.button("전체선택", key="adv_select_all"):
                st.session_state["selected_ids"] = df_stats["video_id"].tolist(); st.rerun()
            if csel2.button("전체해제", key="adv_clear_all"):
                st.session_state["selected_ids"] = []; st.rerun()
            st.markdown("---")
            if st.session_state["mode"] == "검색 모드":
                st.subheader("➕ 추가 URL/ID 병합")
                add_text = st.text_area("추가할 URL/ID (줄바꿈 구분)", height=100,
                                        placeholder="https://youtu.be/XXXXXXXXXXX\nXXXXXXXXXXX\n...", key="adv_add_text")
                if st.button("추가 병합 실행", key="adv_merge_btn"):
                    add_ids = extract_video_ids_from_text(add_text or "")
                    already = set(df_stats["video_id"].tolist())
                    dup = [v for v in add_ids if v in already]
                    add_ids = [v for v in add_ids if v not in already]
                    if dup: st.info(f"⚠️ 기존 목록과 중복 {len(dup)}개 제외")
                    if add_ids:
                        rt = RotatingYouTube(YT_API_KEYS, log=lambda m: st.write(m))
                        add_stats = yt_video_statistics(rt, add_ids)
                        add_df = pd.DataFrame(add_stats)
                        if not add_df.empty and "publishedAt" in add_df.columns:
                            add_df["publishedAt"] = (
                                pd.to_datetime(add_df["publishedAt"], errors="coerce", utc=True)
                                .dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d %H:%M:%S")
                            )
                        st.session_state["df_stats"] = (
                            pd.concat([st.session_state["df_stats"], add_df], ignore_index=True)
                            .drop_duplicates(subset=["video_id"])
                            .sort_values(by="viewCount", ascending=False)
                            .reset_index(drop=True)
                        )
                        st.session_state["selected_ids"] = list(dict.fromkeys(st.session_state["selected_ids"] + add_ids))
                        st.success(f"추가 {len(add_ids)}개 병합 완료"); st.rerun()
                    else:
                        st.info("추가할 신규 URL/ID가 없습니다.")
            st.markdown("---")
            if st.button("다음: 댓글추출로 이동", type="primary", key="adv_next_to_comments"):
                if not st.session_state["selected_ids"]:
                    st.warning("선택된 영상이 없습니다.")
                else:
                    st.session_state["focus_step"] = 3
                    st.rerun()

    # ③ 댓글추출
    expanded3 = (st.session_state["focus_step"] == 3)
    with st.expander("③ 댓글추출", expanded=expanded3):
        if not st.session_state["selected_ids"]:
            st.info("②에서 먼저 영상을 선택하세요.")
        else:
            st.write(f"대상 영상 수: **{len(st.session_state['selected_ids'])}**")
            include_replies_local = st.checkbox("대댓글 포함(이 단계에서만 적용)",
                                                value=include_replies, key="adv_include_replies_collect")
            if st.button("댓글 수집 시작", type="primary", use_container_width=True, key="adv_collect_btn"):
                if not YT_API_KEYS:
                    st.error("YouTube API Key가 없습니다.")
                else:
                    if not lock_guard_start_or_warn():
                        st.stop()
                    try:
                        df_stats = st.session_state["df_stats"]
                        target_ids = st.session_state["selected_ids"]
                        if df_stats is not None and not df_stats.empty and "viewCount" in df_stats.columns:
                            video_list = df_stats[df_stats["video_id"].isin(target_ids)].sort_values("viewCount", ascending=False).to_dict('records')
                        else:
                            video_list = [{"video_id": vid, "title": "", "shortType": "Clip"} for vid in target_ids]
                        prog = st.progress(0, text="수집 진행")
                        log_ph = st.empty()
                        rows = parallel_collect_comments(
                            video_list=video_list,
                            rt_keys=YT_API_KEYS,
                            include_replies=st.session_state.get("adv_include_replies_collect", False),
                            max_total_comments=MAX_TOTAL_COMMENTS,
                            max_per_video=MAX_COMMENTS_PER_VIDEO,
                            log_callback=log_ph.write,
                            prog_callback=prog.progress
                        )
                        if rows:
                            df_comments = pd.DataFrame(rows).applymap(clean_illegal)
                            st.session_state["df_comments"] = df_comments
                            df_analysis = sample_max_5000(df_comments)
                            st.session_state["df_analysis"] = df_analysis
                            a_text, _, _ = serialize_comments_for_llm(
                                df_analysis, max_rows=1500, max_chars_per_comment=280, max_total_chars=450_000
                            )
                            st.session_state["adv_serialized_sample"] = a_text
                            st.success("댓글 수집 완료! 필요 시 아래에서 파일 다운로드 후 다음 단계로 이동하세요.")
                        else:
                            st.warning("댓글이 수집되지 않았습니다.")
                    finally:
                        release_lock()
                if st.button("다음: AI분석으로 이동", type="primary", key="adv_go_to_step4"):
                    st.session_state["focus_step"] = 4
                    st.rerun()

    # ④ AI분석
    expanded4 = (st.session_state["focus_step"] == 4)
    with st.expander("④ AI분석", expanded=expanded4):
        df_analysis = st.session_state["df_analysis"]
        if df_analysis is None or df_analysis.empty:
            st.info("③에서 댓글 수집을 완료하면 여기에 분석 기능이 활성화됩니다.")
        else:
            st.write(f"분석 대상 샘플 댓글 수: **{len(df_analysis):,}** (최대 5,000개)")
            analysis_keyword = st.text_input("관련 키워드(분석 컨텍스트)",
                                             value=st.session_state.get("last_keyword", ""),
                                             placeholder="예: 윤두준", key="adv_analysis_keyword")
            user_question_adv = st.text_area("사용자 질문", height=80, placeholder="예: 최근 반응은?", key="adv_user_question")

            if st.button("✨ AI 분석 실행", type="primary", key="adv_run_gem"):
                if not GEMINI_API_KEYS:
                    st.error("Gemini API Key가 없습니다.")
                else:
                    if not lock_guard_start_or_warn():
                        st.stop()
                    try:
                        append_log("고급", analysis_keyword, user_question_adv)  # no-op
                        st.session_state["adv_history"] = []
                        st.session_state["adv_followups"] = []
                        a_text = st.session_state.get("adv_serialized_sample", "") or serialize_comments_for_llm(df_analysis)[0]
                        system_instruction = (
                            "너는 유튜브 댓글을 분석하는 어시스턴트다. "
                            "아래 키워드와 댓글 샘플을 바탕으로 사용자 질문에 답하라. "
                            "핵심 포인트를 항목화하고, 대략적 비율과 대표 코멘트(10개미만)도 제시하라. 출력은 한국어. "
                            "키워드가 배우명(사람이름)이면 배우 중심으로 분석하라. "
                            "반드시 모든 댓글을 읽고 답변하라."
                        )
                        user_payload = f"[키워드]: {analysis_keyword}\n[질문]: {user_question_adv}\n\n[댓글 샘플]:\n{a_text}\n"
                        out = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, system_instruction, user_payload)
                        st.session_state["adv_result_text"] = out
                        st.session_state["adv_history"].append((user_question_adv or "최근 반응 요약", out))
                        st.success("AI 분석 완료")
                    finally:
                        release_lock()

            if st.session_state.get("adv_result_text"):
                st.markdown("#### 📄 분석 결과")
                st.markdown(st.session_state["adv_result_text"])
                if st.session_state["adv_followups"]:
                    st.markdown("### 📝 추가 질문 히스토리")
                    for i, (q, a) in enumerate(st.session_state["adv_followups"], start=1):
                        with st.expander(f"Q{i}. {q}", expanded=True):
                            st.markdown(a or "_응답 없음_")
                st.markdown("#### ➕ 추가 질문하기")
                st.text_input("추가 질문", placeholder="예: 긍/부정 키워드 Top5는?",
                              key="adv_follow_q", on_change=handle_followup_advanced)
                st.button("질문 보내기(고급)", key="adv_follow_btn", on_click=handle_followup_advanced)

                if st.button("💾 세션 저장하기", key="adv_save_session_analysis"):
                    name, drive_id = save_current_session(None)
                    if drive_id:
                        st.success(f"세션 저장 완료 · Drive 폴더: https://drive.google.com/drive/folders/{drive_id}")
                    else:
                        st.success(f"세션 저장 완료(로컬) · {name}")

    render_quant_viz(st.session_state.get("df_comments"), st.session_state.get("df_stats"), scope_label="(KST 기준)")
    render_downloads(st.session_state.get("df_comments"), st.session_state.get("df_analysis"),
                     st.session_state.get("df_stats"), prefix=f"adv_{len(st.session_state.get('selected_ids', []))}vids")

    if st.button("💾 세션 저장하기", key="adv_save_session_comments"):
        name, drive_id = save_current_session(None)
        if drive_id:
            st.success(f"세션 저장 완료 · Drive 폴더: https://drive.google.com/drive/folders/{drive_id}")
        else:
            st.success(f"세션 저장 완료(로컬) · {name}")

# ===================== 3) 세션 아카이브 =====================
with tab_sessions:
    st.subheader("저장된 세션 아카이브")

    drive_ok = bool(GDRIVE_PARENT_FOLDER_ID and GDRIVE_KEYS)
    if not drive_ok:
        st.info("Drive 설정이 없어 로컬 세션만 표시됩니다.")
        sess_list = list_sessions_local()
        if not sess_list:
            st.info("저장된 세션이 없습니다.")
        else:
            selected = st.selectbox("세션 선택(로컬)", sess_list, key="sess_select_local")
            sess_path = os.path.join(SESS_DIR, selected)
            qa_file = os.path.join(sess_path, "qa.json")
            if os.path.exists(qa_file):
                with open(qa_file, encoding="utf-8") as f:
                    qa_data = json.load(f)
                st.write("### 질문/응답")
                for i,(q,a) in enumerate(qa_data.get("simple_history",[]),1):
                    with st.expander(f"[심플 Q{i}] {q}", expanded=False):
                        st.markdown(a)
                for i,(q,a) in enumerate(qa_data.get("adv_history",[]),1):
                    with st.expander(f"[고급 Q{i}] {q}", expanded=False):
                        st.markdown(a)
            if st.button("📦 ZIP 만들기/새로고침", key="sess_zip_build_local"):
                zip_session(selected); st.success("ZIP 생성/갱신 완료")
            zip_path = os.path.join(SESS_DIR, f"{selected}.zip")
            if os.path.exists(zip_path):
                with open(zip_path, "rb") as f:
                    st.download_button("⬇️ 세션 전체 다운로드 (ZIP)", data=f.read(), file_name=f"{selected}.zip")
            st.write("### 세션 폴더 파일 (CSV/JSON/PNG)")
            for fn in sorted(os.listdir(sess_path)):
                p = os.path.join(sess_path, fn)
                if os.path.isfile(p):
                    with open(p, "rb") as f:
                        st.download_button(f"⬇️ {fn}", data=f.read(), file_name=fn, key=f"dl_{selected}_{fn}")
    else:
        # Drive 세션 전체 목록 표시
        try:
            rd = RotatingDrive(GDRIVE_KEYS, log=lambda m: st.write(m))
            folders = drive_list_folders(rd, GDRIVE_PARENT_FOLDER_ID)
            if not folders:
                st.info("Drive에 저장된 세션이 없습니다.")
            else:
                options = [f["name"] for f in folders]
                idx_map = {f["name"]: f["id"] for f in folders}
                selected_name = st.selectbox("세션 선택(Drive)", options, key="sess_select_drive")
                selected_id = idx_map.get(selected_name)
                if selected_id:
                    files = drive_list_files_in_folder(rd, selected_id)
                    manifest = next((x for x in files if x["name"] == "manifest.json"), None)
                    if manifest:
                        st.markdown(f"- **Drive 폴더 링크:** https://drive.google.com/drive/folders/{selected_id}")
                    st.write("### 파일 목록 (Drive)")
                    for fobj in files:
                        name = fobj.get("name")
                        size = fobj.get("size", "0")
                        vlink = fobj.get("webViewLink")
                        dlink = fobj.get("webContentLink")
                        mt = fobj.get("mimeType","")
                        created = fobj.get("createdTime","")
                        col1, col2 = st.columns([6,4])
                        with col1:
                            st.markdown(f"**{name}**  \n타입: `{mt}` · 생성: `{created}` · 크기: {size}")
                        with col2:
                            if vlink:
                                st.link_button("열기", vlink, help="브라우저에서 보기", key=f"view_{selected_id}_{name}")
                            if dlink:
                                st.link_button("다운로드", dlink, help="바로 다운로드", key=f"dl_{selected_id}_{name}")
                    st.caption("※ Drive 링크는 접근 권한이 있는 사용자만 열 수 있습니다.")
        except Exception as e:
            st.warning(f"Drive 아카이브 로드 실패: {e}")

# ===================== 초기화 버튼 =====================
st.markdown("---")
if st.button("🔄 초기화 하기", type="secondary"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()
