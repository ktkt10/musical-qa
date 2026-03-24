# 演習：RAGを活用したQAシステム構築（LangChain）
# 8. 実践演習: QAシステムの構築とデプロイ
# ============================================================
# ミュージカル QA アプリ (Streamlit + LangChain + FAISS)
# ============================================================
#【概要】
# Wikipediaの日本語記事（ブロードウェイ・ロンドン・ウィーンの
# 代表的なミュージカル21作品）をデータソースとしたQAアプリ
#
# ① Wikipedia から各作品の記事を取得
#          ↓
# ② テキストをチャンクに分割し、OpenAI Embeddings でベクター化
#          ↓
# ③ FAISS ベクターストアに保存（初回のみ / 2回目以降はロード）
#          ↓
# ④ ユーザーが質問を入力
#          ↓
# ⑤ 質問に関連するチャンクを FAISS で検索（RAG）
#          ↓
# ⑥ GPT-4o-mini が検索結果をもとに回答を生成
#          ↓
# ⑦ Streamlit 画面に回答を表示
#
#【実行方法】
# streamlit run musical-qa.py
#  初回のみWikipedia取得 + Embeddings変換 (数分かかる)
#
# ライブラリのインストール (wikipediaが必要)
# pip install streamlit langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu wikipedia openai
# または pip install -r requirements.txt


# 1) ライブラリのインポート
import os
import streamlit as st

# Streamlit CloudのSecretsから取得、なければ環境変数から取得
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass  # ローカル環境では~/.zshrcの環境変数をそのまま使用
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ============================================================
# 2) ミュージカル作品リスト
# ============================================================
MUSICAL_TITLES = [
    # ブロードウェイ
    "キャッツ (ミュージカル)",
    "オペラ座の怪人 (ミュージカル)",
    "レ・ミゼラブル (ミュージカル)",
    "ミス・サイゴン",
    "ライオンキング (ミュージカル)",
    "シカゴ (ミュージカル)",
    "ウィキッド",
    "ハミルトン (ミュージカル)",
    "ディア・エヴァン・ハンセン",
    "ムーラン・ルージュ! ザ・ミュージカル",
    "スウィーニー・トッド",
    "アニー (ミュージカル)",
    # ロンドン（ウエストエンド）
    "メリー・ポピンズ (ミュージカル)",
    "ビリー・エリオット (ミュージカル)",
    "マチルダ (ミュージカル)",
    "ジーザス・クライスト・スーパースター",
    "エビータ",
    "雨に唄えば",
    # ウィーン
    "エリザベート (ミュージカル)",
    "モーツァルト! (ミュージカル)",
    "レベッカ (ミュージカル)",
    "マリー・アントワネット (ミュージカル)",
]

VECTOR_STORE_PATH = "musical_vector_store"

# ============================================================
# 3) ベクターストアの作成またはロード
# ============================================================
@st.cache_resource(show_spinner=False)
def load_vector_store():
    embeddings = OpenAIEmbeddings()

    if os.path.exists(VECTOR_STORE_PATH):
        # 既存のベクターストアをロード
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        # 初回作成時に保存した失敗リストを読み込む
        failed_path = os.path.join(VECTOR_STORE_PATH, "failed_titles.txt")
        if os.path.exists(failed_path):
            with open(failed_path, "r") as f:
                failed = [line.strip() for line in f if line.strip()]
        else:
            failed = []
        return vector_store, failed, False  # False = ロード（初回作成ではない）

    # ベクターストアが存在しない場合は作成
    all_docs = []
    failed_titles = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, title in enumerate(MUSICAL_TITLES):
        status_text.text(f"取得中: {title} ({i+1}/{len(MUSICAL_TITLES)})")
        try:
            loader = WikipediaLoader(query=title, load_max_docs=1, lang="ja")
            docs = loader.load()
            # 内容が薄い記事は除外
            valid_docs = [d for d in docs if len(d.page_content) > 100]
            if valid_docs:
                all_docs.extend(valid_docs)
            else:
                failed_titles.append(title)
        except Exception as e:
            failed_titles.append(title)

        progress_bar.progress((i + 1) / len(MUSICAL_TITLES))

    status_text.text("テキストを分割してベクターストアを作成中...")

    # テキストをチャンクに分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = text_splitter.split_documents(all_docs)

    # ベクターストアを作成して保存
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)

    # 失敗リストをファイルに保存（次回ロード時に使用）
    failed_path = os.path.join(VECTOR_STORE_PATH, "failed_titles.txt")
    with open(failed_path, "w") as f:
        f.write("\n".join(failed_titles))

    progress_bar.empty()
    status_text.empty()

    return vector_store, failed_titles, True  # True = 初回作成


# ============================================================
# 4) Streamlit UI
# ============================================================
st.title("🎭 ミュージカル QA アプリ")
st.caption("ブロードウェイ・ロンドン・ウィーンの代表的なミュージカルについて質問できます")

# ベクターストアのロード
with st.spinner("データを準備中..."):
    vector_store, failed_titles, is_new = load_vector_store()

if is_new and failed_titles:
    with st.expander("⚠️ 取得できなかった作品"):
        for t in failed_titles:
            st.write(f"- {t}")

st.success("✅ Wikipediaデータの読み込みが完了しました。質問をどうぞ！")

# 対応作品一覧の表示（読み込み失敗した作品を除外）
loaded_titles = [t for t in MUSICAL_TITLES if t not in (failed_titles or [])]
with st.expander("📋 対応ミュージカル一覧"):
    st.info("💡 一覧にない作品についても、AIの知識をもとに回答できます。")
    col1, col2, col3 = st.columns(3)
    broadway = [t for t in MUSICAL_TITLES[:12] if t in loaded_titles]
    london   = [t for t in MUSICAL_TITLES[12:18] if t in loaded_titles]
    vienna   = [t for t in MUSICAL_TITLES[18:] if t in loaded_titles]
    with col1:
        st.markdown("**ブロードウェイ**")
        for t in broadway:
            st.write(f"・{t.replace(' (ミュージカル)', '')}")
    with col2:
        st.markdown("**ロンドン**")
        for t in london:
            st.write(f"・{t.replace(' (ミュージカル)', '')}")
    with col3:
        st.markdown("**ウィーン**")
        for t in vienna:
            st.write(f"・{t.replace(' (ミュージカル)', '')}")

# ============================================================
# 5) QA チェーンの初期化と質問処理
# ============================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# プロンプトテンプレート
# 回答の先頭に情報源タグを付けるよう指示する
prompt = ChatPromptTemplate.from_template("""
以下のコンテキストをもとに質問に答えてください。
コンテキストに十分な情報がある場合はそれを使い、ない場合はあなた自身の知識で答えてください。
回答の先頭に、必ず以下のどちらかのタグを付けてください：
- コンテキストを使用した場合：[データベース参照]
- 自身の知識を使用した場合：[AI知識]

コンテキスト: {context}

質問: {question}
""")

# コンテキストを整形する関数
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LCEL（LangChain Expression Language）でチェーンを構築
qa_chain = (
    {
        "context": vector_store.as_retriever(search_kwargs={"k": 3}) | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 質問入力
user_input = st.text_input("質問を入力してください:", placeholder="例：エリザベートはどんなストーリーですか？")

if user_input:
    with st.spinner("回答を生成中..."):
        answer = qa_chain.invoke(user_input)

    # 情報源タグを検出して表示を切り替える
    if answer.startswith("[データベース参照]"):
        st.success("📚 情報源：ミュージカルデータベース（FAISS）")
        answer_text = answer.replace("[データベース参照]", "").strip()
    elif answer.startswith("[AI知識]"):
        st.info("🤖 情報源：AI の知識（GPT-4o-mini）")
        answer_text = answer.replace("[AI知識]", "").strip()
    else:
        answer_text = answer

    st.markdown("### 回答")
    st.write(answer_text)
