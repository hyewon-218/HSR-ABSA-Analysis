"""Share of empathy evidence excerpts containing a price-related term (Section 3.1).
Usage (from the repository root): python analysis/05_empathy_price_terms.py"""
import re, pandas as pd
d = pd.read_csv('data/claude_sonnet46_annotations.csv', encoding='utf-8-sig')
PAT = re.compile(r'price|fare|expensive|cheap|cost|money|euro|€|\$|preis|teuer|billig|günstig|prezz|costos|caro|prix|cher|precio|barato|'
                 r'料金|値段|高い|安い|价格|价钱|票价|贵|便宜|價格|票價|貴|가격|요금|비싸|저렴|운임', re.I)
m = d.Empathy_sentiment != 'Not_mentioned'
share = d.loc[m, 'Empathy_evidence'].fillna('').astype(str).str.contains(PAT).mean()
print(f'empathy excerpts: {m.sum()}; with a price-related term: {100 * share:.1f}%')
