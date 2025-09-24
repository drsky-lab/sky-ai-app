# SKY AI

Robust calendar assistant with hidden profile support and safer web sourcing.

---

## Reference

**Ref ID**: SKY-AI/2025-09-24/CHAT-001  
**Title**: SKY AI — Robust Calendar + Hidden Profile + Clean Sources  
**Date (IST)**: September 24, 2025  

### Highlights
- Robust calendar handler (IST, supports “date before yesterday”, “N days ago”, typos).  
- Embedded hidden creator profile in system context (used only when asked).  
- Auto web search (Tavily) only when needed; removed unrelated sources.  
- Sanitizers strip unsolicited intros/links; contact block only on complex asks.  
- Simple date-image generator (Pillow).  
- Footer: *“Built by Shashikant Yadav • SKY AI”*.  

**Files touched**:  
- `app.py`  
- `requirements.txt`  
- `.streamlit/secrets.toml` (keys: `hf_api_key`, optional `tavily_api_key`)  

---

## Commit Message (suggested)

