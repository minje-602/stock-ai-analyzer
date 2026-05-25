import json
import os
import yfinance as yf
import pandas as pd
from datetime import datetime

PORTFOLIO_FILE = "portfolio.json"
INITIAL_CASH = 10_000_000  # 초기 자금 1000만원

def load_portfolio():
    """포트폴리오 파일 로드. 없으면 초기 상태 생성"""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {
        "cash": INITIAL_CASH,
        "holdings": {},
        "history": []
    }

def save_portfolio(portfolio):
    """포트폴리오 파일 저장"""
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, ensure_ascii=False, indent=2)

def get_current_price(ticker):
    """실시간(최근) 종가 조회"""
    try:
        df = yf.download(ticker, period="1d", interval="1m", auto_adjust=True, progress=False)
        if df.empty:
            df = yf.download(ticker, period="5d", interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except:
        return None

def buy(portfolio, ticker, name, qty, price):
    """매수: 현금 차감 + 보유종목 갱신(평단가 재계산)"""
    cost = qty * price
    if cost > portfolio["cash"]:
        return False, "현금이 부족합니다."

    portfolio["cash"] -= cost

    if ticker in portfolio["holdings"]:
        h = portfolio["holdings"][ticker]
        total_qty = h["qty"] + qty
        total_cost = h["avg_price"] * h["qty"] + price * qty
        h["avg_price"] = total_cost / total_qty
        h["qty"] = total_qty
    else:
        portfolio["holdings"][ticker] = {
            "name": name, "qty": qty, "avg_price": price
        }

    portfolio["history"].append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "매수", "ticker": ticker, "name": name,
        "qty": qty, "price": price, "amount": cost
    })
    return True, f"{name} {qty}주 매수 완료"

def sell(portfolio, ticker, qty, price):
    """매도: 보유수량 차감 + 현금 증가"""
    if ticker not in portfolio["holdings"]:
        return False, "보유하지 않은 종목입니다."

    h = portfolio["holdings"][ticker]
    if qty > h["qty"]:
        return False, f"보유 수량({h['qty']}주)보다 많습니다."

    revenue = qty * price
    portfolio["cash"] += revenue
    name = h["name"]

    h["qty"] -= qty
    if h["qty"] == 0:
        del portfolio["holdings"][ticker]

    portfolio["history"].append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "매도", "ticker": ticker, "name": name,
        "qty": qty, "price": price, "amount": revenue
    })
    return True, f"{name} {qty}주 매도 완료"

def evaluate(portfolio):
    """포트폴리오 평가: 총 자산, 평가손익 계산"""
    total_eval = portfolio["cash"]
    holdings_detail = []

    for ticker, h in portfolio["holdings"].items():
        cur_price = get_current_price(ticker)
        if cur_price is None:
            cur_price = h["avg_price"]
        eval_amount = cur_price * h["qty"]
        cost_amount = h["avg_price"] * h["qty"]
        profit = eval_amount - cost_amount
        profit_rate = (profit / cost_amount * 100) if cost_amount > 0 else 0

        holdings_detail.append({
            "ticker": ticker, "name": h["name"], "qty": h["qty"],
            "avg_price": h["avg_price"], "cur_price": cur_price,
            "eval_amount": eval_amount, "profit": profit, "profit_rate": profit_rate
        })
        total_eval += eval_amount

    total_profit = total_eval - INITIAL_CASH
    total_profit_rate = (total_profit / INITIAL_CASH) * 100

    return {
        "total_eval": total_eval,
        "cash": portfolio["cash"],
        "total_profit": total_profit,
        "total_profit_rate": total_profit_rate,
        "holdings": holdings_detail
    }

def reset_portfolio():
    """포트폴리오 초기화"""
    portfolio = {
        "cash": INITIAL_CASH,
        "holdings": {},
        "history": []
    }
    save_portfolio(portfolio)
    return portfolio