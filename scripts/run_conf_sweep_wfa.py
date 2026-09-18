"""
[Lumos] 어제 신뢰도 Sweep 모델 WFA 검증
=========================================
어제 run_doe_confidence_sweep.py 의 모델 구조를 완전 동일하게 유지하되
주간 롤링 WFA로 미래참조 완전 차단:

[동일하게 유지]
 - CrossAsset Veto (CrossAssetDislocationModel z=1.6)
 - SOXX + TQQQ 60분봉 EMA20 추세 필터
 - Screen 3: 15분봉 VWAP_Diff<=1.5 & RSI_14<=62.0 & BB_Lower 눌림목
 - TP=+3.5% / SL=-2.0% (5분봉 Path Dissection, 18봉=90분)
 - Intra-bar Illusion 해소 로직
 - 3-Out 일일 서킷 브레이커
 - 14:30 진입 컷오프, 15:45 EOD 청산

[WFA 추가]
 - 학습: 직전 104주(2년) 슬라이딩 윈도우
 - 테스트: 다음 1주 (완전 블라인드 OOS)
 - Conf 스윕: 0.60 / 0.62 / 0.64
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_lake import MarketDataLake
from core.ml_engine import MLFeatureEngine
from core.heterogeneous_models import CrossAssetDislocationModel

ROLL_WKS  = 104
INIT_CAP  = 10_000_000.0
TP_PCT    = 0.035
SL_PCT    = 0.020
SLIP      = 0.03        # 달러 슬리피지 (원본 그대로 $0.03)
FEE_RATE  = 0.0020
TS_BARS   = 18          # 5분봉 90분
CUT_ENTRY = "14:30"
CUT_EOD   = "15:45"
CONFS     = [0.60, 0.62, 0.64]


def precompute_cross_dirs(tqqq_15m, nvda_map, soxx_map, qqq_map, vix_map):
    """CrossAsset 방향 전체 사전계산 (원본 동일)"""
    cross_mod = CrossAssetDislocationModel(dislocation_z_threshold=1.6)
    dts   = tqqq_15m["datetime"].values
    cls_  = tqqq_15m["Close"].values
    dirs  = ["NONE"] * len(tqqq_15m)
    confs = [0.50]   * len(tqqq_15m)
    for i in range(len(tqqq_15m)):
        if i < 5: continue
        cur_t  = dts[i]; past_t = dts[i-5]
        if not (cur_t in nvda_map and past_t in nvda_map and
                cur_t in qqq_map  and past_t in qqq_map  and
                cur_t in vix_map  and past_t in vix_map):
            continue
        s_r  = float(cls_[i] / cls_[i-5] - 1.0)
        n_r  = float(nvda_map[cur_t] / nvda_map[past_t] - 1.0)
        sx_r = float(soxx_map[cur_t] / soxx_map[past_t] - 1.0) if (cur_t in soxx_map and past_t in soxx_map) else n_r
        q_r  = float(qqq_map[cur_t] / qqq_map[past_t] - 1.0)
        v_r  = float(vix_map[cur_t] / vix_map[past_t] - 1.0)
        try:
            code, cf, _ = cross_mod.predict_signal(tqqq_ret=s_r, nvda_ret=n_r, soxx_ret=sx_r, qqq_ret=q_r, vix_ret=v_r, tnx_ret=0.0)
            dirs[i]  = "LONG_TQQQ" if code > 0 else ("SHORT_SQQQ" if code < 0 else "NONE")
            confs[i] = cf
        except Exception:
            pass
    return dirs, confs


def run_conf_wfa(df_feat, df_tqqq_feat_idx, df_sqqq_feat_idx,
                 tqqq_5m_d, sqqq_5m_d,
                 soxx_60m, tqqq_60m,
                 unique_weeks, fcols, conf_thr, label):
    """단일 conf 임계값으로 WFA 시뮬레이션"""
    print(f"\n{'='*90}\n[{label}] conf>={conf_thr*100:.0f}% WFA 시작\n{'='*90}")
    t0   = time.time()
    cap  = INIT_CAP; peak = INIT_CAP; mdd = 0.0
    trd  = []; tid = 0; n_wks = len(unique_weeks) - ROLL_WKS

    for wi in range(ROLL_WKS, len(unique_weeks)):
        tw  = unique_weeks[wi]
        trw = unique_weeks[wi-ROLL_WKS:wi]

        # ── 학습 ──
        mask  = df_feat["week_id"].isin(trw)
        dtr   = df_feat[mask].copy()
        # Triple Barrier 라벨 (기본 +3%/-2%)
        n = len(dtr)
        lbl = np.zeros(n, int)
        H  = dtr["High"].values; Lo = dtr["Low"].values; C = dtr["Close"].values
        for i in range(n):
            p0 = C[i]
            if p0 <= 0: continue
            ei = min(i+7, n)
            for j in range(i+1, ei):
                hr = (H[j]-p0)/p0; lr = (Lo[j]-p0)/p0
                if hr >= 0.030 and lr > -0.020: lbl[i]=1; break
                elif lr <= -0.030 and hr < 0.020: lbl[i]=-1; break
                elif (hr>=0.030 and lr<=-0.020) or (lr<=-0.030 and hr>=0.020): break

        y = pd.Series(lbl).map({1:2,-1:0,0:1}).fillna(1).astype(int)
        X = dtr[fcols].fillna(0.0)
        clf = LGBMClassifier(objective="multiclass", num_class=3, class_weight="balanced",
                             n_estimators=85, max_depth=4, learning_rate=0.03,
                             random_state=42, verbosity=-1, n_jobs=-1)
        clf.fit(X, y)

        # ── OOS 예측 ──
        dts_wk = df_feat[df_feat["week_id"] == tw].copy()
        if dts_wk.empty: continue
        pr = clf.predict_proba(dts_wk[fcols].fillna(0.0))
        p_s, p_n, p_l = pr[:,0], pr[:,1], pr[:,2]
        gbdt_dirs = []; gbdt_confs = []
        for i in range(len(dts_wk)):
            ps, pn, pl = p_s[i], p_n[i], p_l[i]
            if pl > pn and pl > ps:
                gbdt_dirs.append("LONG_TQQQ")
                gbdt_confs.append(min(0.95, max(0.50, 0.50+(pl-0.333)*1.15)))
            elif ps > pn and ps > pl:
                gbdt_dirs.append("SHORT_SQQQ")
                gbdt_confs.append(min(0.95, max(0.50, 0.50+(ps-0.333)*1.15)))
            else:
                gbdt_dirs.append("NONE"); gbdt_confs.append(0.50)
        dts_wk = dts_wk.copy()
        dts_wk["gbdt_dir"]  = gbdt_dirs
        dts_wk["gbdt_conf"] = gbdt_confs
        dts_wk = dts_wk.set_index("datetime", drop=False)

        # ── 일별 시뮬레이션 ──
        for d_str in sorted(dts_wk["date_str"].unique()):
            day15 = dts_wk[dts_wk["date_str"] == d_str]
            slc   = 0
            b_idx = 0; rows = list(day15.iterrows()); n_bars = len(rows)

            while b_idx < n_bars:
                dt_idx, row = rows[b_idx]
                cur_dt  = row["datetime"]
                ts      = row["time_str"]

                # 14:30 컷오프
                if b_idx < 1 or ts > CUT_ENTRY:
                    b_idx += 1; continue
                # 3-out 서킷 브레이커
                if slc >= 3:
                    b_idx += 1; continue

                gbdt_dir  = row["gbdt_dir"]
                gbdt_conf = float(row["gbdt_conf"])
                cross_dir = row.get("cross_dir", "NONE")

                # ─ GBDT 기준 미달 ─
                if gbdt_dir not in ("LONG_TQQQ","SHORT_SQQQ") or gbdt_conf < conf_thr:
                    b_idx += 1; continue

                # ─ CrossAsset Veto ─
                if ((gbdt_dir=="LONG_TQQQ"  and cross_dir=="SHORT_SQQQ") or
                    (gbdt_dir=="SHORT_SQQQ" and cross_dir=="LONG_TQQQ")):
                    b_idx += 1; continue

                # ─ Screen 1: 60분봉 EMA20 추세 ─
                past60s = soxx_60m[soxx_60m["datetime"] <= cur_dt]
                past60l = tqqq_60m[tqqq_60m["datetime"] <= cur_dt]
                if len(past60s) >= 20 and len(past60l) >= 20:
                    soxx_c  = past60s["Close"].iloc[-1]
                    tqqq_c  = past60l["Close"].iloc[-1]
                    soxx_e  = past60s["ema20"].iloc[-1]
                    tqqq_e  = past60l["ema20"].iloc[-1]
                    if gbdt_dir == "LONG_TQQQ":
                        if not (soxx_c >= soxx_e*0.998 and tqqq_c >= tqqq_e*0.998):
                            b_idx += 1; continue
                    else:
                        if not (soxx_c <= soxx_e*1.002):
                            b_idx += 1; continue

                # ─ Screen 3: 15분봉 눌림목 (VWAP/RSI/BB) ─
                if gbdt_dir == "LONG_TQQQ":
                    vd   = float(row.get("VWAP_Diff", 0.0))
                    rsi  = float(row.get("RSI_14", 50.0))
                    bbl  = float(row.get("BB_Lower", 0.0))
                    cc   = float(row["Close"])
                    dip  = (vd <= 1.5) and (rsi <= 62.0)
                    if bbl > 0: dip = dip and (cc >= bbl*1.001)
                else:
                    if cur_dt in df_sqqq_feat_idx.index:
                        rs   = df_sqqq_feat_idx.loc[cur_dt]
                        vd   = float(rs.get("VWAP_Diff", 0.0))
                        rsi  = float(rs.get("RSI_14", 50.0))
                        bbl  = float(rs.get("BB_Lower", 0.0))
                        cc   = float(rs["Close"])
                        dip  = (vd <= 1.5) and (rsi <= 62.0)
                        if bbl > 0: dip = dip and (cc >= bbl*1.001)
                    else:
                        dip = False
                if not dip:
                    b_idx += 1; continue

                # ─ 진입 ─
                sym = "TQQQ" if gbdt_dir=="LONG_TQQQ" else "SQQQ"
                if sym == "TQQQ":
                    base_px = float(row["Close"])
                else:
                    if cur_dt in df_sqqq_feat_idx.index:
                        base_px = float(df_sqqq_feat_idx.loc[cur_dt]["Close"])
                    else:
                        b_idx += 1; continue
                entry_px = round(base_px + SLIP, 2)
                shares   = int(cap / entry_px)
                invested = shares * entry_px
                if shares <= 0 or invested <= 0:
                    b_idx += 1; continue

                # ─ 5분봉 Path Dissection ─
                day5 = tqqq_5m_d.get(d_str, pd.DataFrame()) if sym=="TQQQ" else sqqq_5m_d.get(d_str, pd.DataFrame())
                if day5.empty:
                    b_idx += 1; continue
                post5 = day5[day5["datetime"] > cur_dt]
                if post5.empty:
                    b_idx += 1; continue

                tp_px  = round(entry_px*(1+TP_PCT), 2)
                sl_px  = round(entry_px*(1-SL_PCT), 2)
                eval5  = post5.iloc[:TS_BARS]
                ex_px  = None; ex_reason = "TIMESTOP"; ex_ts = ""

                for _, c5 in eval5.iterrows():
                    t5  = c5["time_str"]
                    c5h = float(c5["High"]); c5l = float(c5["Low"]); c5c = float(c5["Close"]); c5o = float(c5["Open"])
                    htp = c5h >= tp_px; hsl = c5l <= sl_px
                    if htp and hsl:
                        ex_px = round((tp_px if c5c>=c5o else sl_px) - SLIP, 2)
                        ex_reason = "TP" if c5c>=c5o else "SL"; ex_ts=t5; break
                    elif htp:
                        ex_px = round(tp_px - SLIP, 2); ex_reason = "TP"; ex_ts=t5; break
                    elif hsl:
                        ex_px = round(sl_px - SLIP, 2); ex_reason = "SL"; ex_ts=t5; break
                    elif t5 >= CUT_EOD:
                        ex_px = round(c5c - SLIP, 2); ex_reason = "EOD"; ex_ts=t5; break
                if ex_px is None:
                    last = eval5.iloc[-1]
                    ex_px = round(float(last["Close"]) - SLIP, 2)
                    ex_ts = last["time_str"]

                raw_ret  = (ex_px / entry_px) - 1.0
                fee      = FEE_RATE
                net_ret  = raw_ret - fee
                pnl      = cap * net_ret
                cap     += pnl
                if cap > peak: peak = cap
                dd = (peak-cap)/peak*100
                if dd > mdd: mdd = dd
                if net_ret <= -0.015: slc += 1
                tid += 1
                trd.append({"tid":tid,"dt":cur_dt,"sym":sym,"ep":round(entry_px,4),
                             "xp":round(ex_px,4),"reason":ex_reason,"ret":round(net_ret*100,3),
                             "pnl":int(round(pnl)),"cap":int(round(cap)),
                             "win":1 if net_ret>0 else 0,"conf":round(gbdt_conf,3)})
                b_idx += 1

        wn = wi-ROLL_WKS+1
        if wn % 30 == 0 or wn == n_wks:
            print(f"   [{wn}/{n_wks}주] {tw} | 거래:{len(trd):,} | 잔고:{int(cap):,}원")

    el = round(time.time()-t0,1)
    tr = (cap/INIT_CAP-1)*100
    print(f"\n✅ [{label}] {el}s | 거래:{len(trd):,} | {tr:+.1f}% | MDD:{mdd:.1f}%")
    return {"label":label,"conf":conf_thr,"final":int(cap),"ret":round(tr,2),
            "mdd":round(mdd,2),"n":len(trd),"trd":trd}


def summarize(res):
    if not res["trd"]: return res
    df = pd.DataFrame(res["trd"])
    df["year"] = df["dt"].str[:4]
    wins = df[df["ret"]>0]; loss = df[df["ret"]<=0]
    gp = wins["ret"].sum(); gl = abs(loss["ret"].sum())
    res["wr"] = round(len(wins)/len(df)*100,2)
    res["pf"] = round(gp/gl,3) if gl>0 else 99.0
    res["exit_dist"] = df["reason"].value_counts().to_dict()
    yrs=[]; ys=INIT_CAP
    for y,g in df.groupby("year"):
        ye=float(g["cap"].iloc[-1])
        gw=g[g["ret"]>0]["ret"].sum(); gl2=abs(g[g["ret"]<=0]["ret"].sum())
        yrs.append({"year":y,"ret":round((ye/ys-1)*100,1),"n":len(g),
                    "wr":round(len(g[g["ret"]>0])/len(g)*100,1),
                    "pf":round(gw/gl2,3) if gl2>0 else 99.0,
                    "L":int((g["sym"]=="TQQQ").sum()),"S":int((g["sym"]=="SQQQ").sum())})
        ys=ye
    res["yearly"]=yrs
    return res


def san(o):
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,(np.floating,)): return float(o)
    if isinstance(o,dict): return {k:san(v) for k,v in o.items()}
    if isinstance(o,list): return [san(i) for i in o]
    return o


def main():
    lake = MarketDataLake()
    ml   = MLFeatureEngine()

    print("="*90+"\n📦 데이터 로드 중...\n"+"="*90)
    tqqq_15m = lake.load_candles("TQQQ","15m").sort_values("datetime").reset_index(drop=True)
    sqqq_15m = lake.load_candles("SQQQ","15m").sort_values("datetime").reset_index(drop=True)
    tqqq_5m  = lake.load_candles("TQQQ","5m").sort_values("datetime").reset_index(drop=True)
    sqqq_5m  = lake.load_candles("SQQQ","5m").sort_values("datetime").reset_index(drop=True)
    soxx_60m = lake.load_candles("SOXX","60m").sort_values("datetime").reset_index(drop=True)
    tqqq_60m = lake.load_candles("TQQQ","60m").sort_values("datetime").reset_index(drop=True)
    soxx_15m = lake.load_candles("SOXX","15m").sort_values("datetime").reset_index(drop=True)
    nvda_15m = lake.load_candles("NVDA","15m").sort_values("datetime").reset_index(drop=True)
    qqq_15m  = lake.load_candles("QQQ", "15m").sort_values("datetime").reset_index(drop=True)
    vix_15m  = lake.load_candles("^VIX","15m").sort_values("datetime").reset_index(drop=True)

    # 60분봉 EMA20 사전계산
    soxx_60m["ema20"] = soxx_60m["Close"].ewm(span=20,adjust=False).mean()
    tqqq_60m["ema20"] = tqqq_60m["Close"].ewm(span=20,adjust=False).mean()

    # datetime 태깅
    for df in [tqqq_15m,sqqq_15m,soxx_60m,tqqq_60m,nvda_15m,qqq_15m,vix_15m,soxx_15m]:
        df["datetime_dt"] = pd.to_datetime(df["datetime"])
        df["date_str"]    = df["datetime_dt"].dt.strftime("%Y-%m-%d")
        df["time_str"]    = df["datetime_dt"].dt.strftime("%H:%M")
    for df in [tqqq_5m,sqqq_5m]:
        df["datetime_dt"] = pd.to_datetime(df["datetime"])
        df["date_str"]    = df["datetime_dt"].dt.strftime("%Y-%m-%d")
        df["time_str"]    = df["datetime_dt"].dt.strftime("%H:%M")

    # CrossAsset 방향 사전계산
    print("⏳ CrossAsset 방향 사전계산...")
    nvda_map = nvda_15m.set_index("datetime")["Close"].to_dict()
    soxx_map = soxx_15m.set_index("datetime")["Close"].to_dict()
    qqq_map  = qqq_15m.set_index("datetime")["Close"].to_dict()
    vix_map  = vix_15m.set_index("datetime")["Close"].to_dict()
    ca_dirs, ca_confs = precompute_cross_dirs(tqqq_15m, nvda_map, soxx_map, qqq_map, vix_map)

    # TQQQ 피처 추출
    print("⏳ 피처 추출...")
    df_feat = ml.extract_features(tqqq_15m)
    df_feat["MACD_Pct"]        = (df_feat["MACD"]/(df_feat["Close"]+1e-9))*100
    df_feat["MACD_Signal_Pct"] = (df_feat["MACD_Signal"]/(df_feat["Close"]+1e-9))*100
    df_feat["MACD_Hist_Pct"]   = (df_feat["MACD_Hist"]/(df_feat["Close"]+1e-9))*100
    df_feat["ATR_Pct"]         = (df_feat["ATR_14"]/(df_feat["Close"]+1e-9))*100
    df_feat["cross_dir"]  = ca_dirs[:len(df_feat)]
    df_feat["cross_conf"] = ca_confs[:len(df_feat)]
    df_feat["datetime_dt"] = pd.to_datetime(df_feat["datetime"])
    df_feat["date_str"] = df_feat["datetime_dt"].dt.strftime("%Y-%m-%d")
    df_feat["time_str"] = df_feat["datetime_dt"].dt.strftime("%H:%M")
    df_feat["week_id"]  = (df_feat["datetime_dt"].dt.isocalendar().year.astype(str)+"-"+
                           df_feat["datetime_dt"].dt.isocalendar().week.astype(str).str.zfill(2))

    # SQQQ 피처 (Screen3용)
    df_sqqq_feat = ml.extract_features(sqqq_15m)
    df_sqqq_feat_idx = df_sqqq_feat.set_index("datetime", drop=False)

    # 피처 컬럼 선별
    EX = {"open","high","low","close","volume","Open","High","Low","Close","Volume",
          "datetime","datetime_dt","date_str","time_str","week_id","cross_dir","cross_conf",
          "date","EMA_9","EMA_21","EMA_50","EMA_200",
          "BB_Upper","BB_Lower","KC_Upper","KC_Lower","ATR_14","MACD","MACD_Signal","MACD_Hist"}
    fcols = [c for c in df_feat.columns if c not in EX and pd.api.types.is_numeric_dtype(df_feat[c])]
    print(f"📊 피처: {len(fcols)}개")

    # 5분봉 날짜별 딕셔너리
    tqqq_5m_d = {d:g for d,g in tqqq_5m.groupby("date_str")}
    sqqq_5m_d = {d:g for d,g in sqqq_5m.groupby("date_str")}

    # 주차 목록
    wks = sorted(df_feat["week_id"].unique())
    n_oos = len(wks) - ROLL_WKS
    print(f"📅 총 {len(wks)}주 | OOS {n_oos}주 ({wks[ROLL_WKS]} ~ {wks[-1]})")

    # 전략별 실행
    all_res = []
    for conf in CONFS:
        lbl = f"Conf≥{int(conf*100)}%"
        res = run_conf_wfa(df_feat, df_feat.set_index("datetime",drop=False),
                           df_sqqq_feat_idx,
                           tqqq_5m_d, sqqq_5m_d,
                           soxx_60m, tqqq_60m,
                           wks, fcols, conf, lbl)
        res = summarize(res); all_res.append(res)

    # 결과 출력
    print("\n\n"+"="*90)
    print("📊 [Conf Sweep WFA 비교 결과] (3중 스크린 + CrossAsset Veto + 5분봉 Path Dissection)")
    print("="*90)
    print(f"{'전략':<15}|{'수익률':>10}|{'MDD':>7}|{'거래':>6}|{'승률':>7}|{'PF':>6}")
    print("-"*60)
    for r in all_res:
        print(f"{r['label']:<15}|{r['ret']:>+9.1f}%|{r['mdd']:>6.1f}%|{r['n']:>5,}|{r['wr']:>6.1f}%|{r['pf']:>6.3f}")

    ays = sorted(set(y["year"] for r in all_res for y in r.get("yearly",[])))
    print(f"\n{'연도':<6}|"+" | ".join(f"{r['label']:<20}" for r in all_res))
    print("-"*80)
    for y in ays:
        row = f"{y:<6}|"
        for r in all_res:
            yd = next((x for x in r.get("yearly",[]) if x["year"]==y), None)
            row += (f"{yd['ret']:>+7.1f}% W{yd['wr']:.0f}% PF{yd['pf']:.2f}" if yd else "N/A".ljust(25))+" | "
        print(row)

    print("\n청산 사유:")
    for r in all_res:
        print(f"  {r['label']}: "+" | ".join(f"{k}={v}" for k,v in sorted(r.get("exit_dist",{}).items())))

    # 저장
    op = PROJECT_ROOT/"data"/"backtest_conf_sweep_wfa.json"
    with open(op,"w",encoding="utf-8") as f:
        json.dump(san({"ts":pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                       "roll":ROLL_WKS,"confs":CONFS,
                       "results":[{k:v for k,v in r.items() if k!="trd"} for r in all_res]}),
                  f,indent=2,ensure_ascii=False)
    for r in all_res:
        pd.DataFrame(r["trd"]).to_csv(
            PROJECT_ROOT/"data"/f"backtest_conf{int(r['conf']*100)}_wfa_trades.csv",
            index=False,encoding="utf-8-sig")
    print(f"\n💾 {op}\n✅ 완료")

if __name__=="__main__":
    main()
