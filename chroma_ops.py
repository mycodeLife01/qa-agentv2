import os
import sys
import json
import chromadb
from chromadb.config import Settings
import redis
from dotenv import load_dotenv

load_dotenv()

# ================= 配置区域 =================
# 如果你的 docker 部署在远程，请修改 HOST
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
RAG_REDIS_URL = os.getenv("RAG_REDIS_URL")
LLAMA_REDIS_CACHE_NAME = os.getenv("LLAMA_REDIS_CACHE_NAME")
LLAMA_DOC_STORE_NAME = os.getenv("LLAMA_DOC_STORE_NAME")
# ===========================================


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


class ChromaAdmin:
    def __init__(self):
        print(
            f"{Colors.HEADER}正在连接 ChromaDB ({CHROMA_HOST}:{CHROMA_PORT})...{Colors.ENDC}"
        )
        try:
            self.client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            self.client.heartbeat()  # 测试连接
            print(f"{Colors.GREEN}✅ 连接成功!{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}❌ 连接失败: {e}{Colors.ENDC}")
            print(f"请检查 Docker 是否运行，端口是否为 {CHROMA_PORT}")
            sys.exit(1)

    def list_collections(self):
        """列出所有集合及其状态"""
        cols = self.client.list_collections()
        if not cols:
            print(f"\n{Colors.WARNING}⚠️  当前数据库为空 (没有 Collection){Colors.ENDC}")
            return None

        print(f"\n{Colors.BOLD}=== Collection 列表 ==={Colors.ENDC}")
        print(f"{'No.':<5} {'Name':<30} {'Count':<10} {'Metadata'}")
        print("-" * 60)

        col_list = []
        for idx, c in enumerate(cols):
            # 获取实时数量
            real_col = self.client.get_collection(c.name)
            count = real_col.count()
            print(
                f"{idx+1:<5} {c.name:<30} {Colors.BLUE}{count:<10}{Colors.ENDC} {c.metadata}"
            )
            col_list.append(c.name)
        return col_list

    def select_collection(self):
        """辅助函数：让用户选择一个集合"""
        cols = self.list_collections()
        if not cols:
            return None

        try:
            choice = input(f"\n请输入集合序号 (1-{len(cols)}) 或输入 0 返回: ")
            if choice == "0":
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(cols):
                return self.client.get_collection(cols[idx])
            else:
                print(f"{Colors.FAIL}无效序号{Colors.ENDC}")
        except ValueError:
            print(f"{Colors.FAIL}请输入数字{Colors.ENDC}")
        return None

    def peek_data(self):
        """查看集合的前几条数据 (用于检查写入格式)"""
        col = self.select_collection()
        if not col:
            return

        print(f"\n{Colors.BOLD}正在读取 {col.name} 的前 5 条数据...{Colors.ENDC}")
        # peek 不会消耗太多资源
        data = col.peek(limit=5)

        if not data["ids"]:
            print(f"{Colors.WARNING}该集合是空的{Colors.ENDC}")
            return

        for i in range(len(data["ids"])):
            print(f"\n{Colors.BLUE}[Document {i+1}]{Colors.ENDC} ID: {data['ids'][i]}")
            print(f"Metadata: {json.dumps(data['metadatas'][i], ensure_ascii=False)}")
            content = data["documents"][i]
            # 如果内容太长，截断显示
            preview = content[:150] + "..." if len(content) > 150 else content
            print(f"Content : {preview}")

    def rag_test(self):
        """核心功能：模拟 RAG 检索"""
        col = self.select_collection()
        if not col:
            return

        while True:
            query = input(f"\n{Colors.BOLD}请输入测试问题 (输入 q 退出): {Colors.ENDC}")
            if query.lower() == "q":
                break

            try:
                k_str = input("Top K (默认3): ")
                k = int(k_str) if k_str.strip() else 3

                print(f"正在检索: '{query}' ...")
                results = col.query(query_texts=[query], n_results=k)

                print(
                    f"\n{Colors.BOLD}=== 检索结果 (Distance 越小越相似) ==={Colors.ENDC}"
                )

                # 解析结果
                ids = results["ids"][0]
                distances = results["distances"][0]
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]

                for i in range(len(ids)):
                    dist = distances[i]
                    # 根据距离显示颜色，帮助判断质量 (假设使用的是 L2 距离)
                    # 如果是 Cosine，这里的逻辑需要反过来，且一般是 > 0.7 算好
                    color = Colors.GREEN if dist < 1.2 else Colors.WARNING

                    print(
                        f"\n{Colors.BOLD}Rank {i+1}{Colors.ENDC} | {color}Distance: {dist:.4f}{Colors.ENDC} | ID: {ids[i]}"
                    )
                    print(f"Metadata: {metadatas[i]}")
                    print(f"Content : {documents[i].strip()}")
                    print("-" * 50)

            except Exception as e:
                print(f"{Colors.FAIL}检索出错: {e}{Colors.ENDC}")

    def delete_collection(self):
        """删除指定集合"""
        col_list = self.list_collections()
        if not col_list:
            return

        name = input(
            f"\n{Colors.FAIL}请输入要【彻底删除】的集合名称 (区分大小写): {Colors.ENDC}"
        )
        if name in col_list:
            confirm = input(f"确认删除 {name}? (y/n): ")
            if confirm.lower() == "y":
                self.client.delete_collection(name)
                print(f"{Colors.GREEN}已删除集合: {name}{Colors.ENDC}")
        else:
            print("取消或名称错误")

    def reset_db(self):
        """重置整个数据库"""
        print(f"\n{Colors.FAIL}{Colors.BOLD}!!! 危险操作 !!!{Colors.ENDC}")
        print("这将清空 ChromaDB 中的所有集合和数据！")
        confirm = input("请输入 'RESET' 确认操作: ")
        if confirm == "RESET":
            try:
                self.client.reset()
                print(f"{Colors.GREEN}数据库已重置{Colors.ENDC}")
            except Exception as e:
                print(
                    f"{Colors.FAIL}重置失败 (可能是服务端 allow_reset=False): {e}{Colors.ENDC}"
                )
                print("尝试手动循环删除所有集合...")
                cols = self.client.list_collections()
                for c in cols:
                    self.client.delete_collection(c.name)
                    print(f"已删除: {c.name}")
        else:
            print("操作已取消")

    def clear_redis_cache(self):
        """清除 Redis 中的 Ingestion Cache 和 DocStore"""
        if not RAG_REDIS_URL:
            print(f"{Colors.FAIL}未配置 RAG_REDIS_URL 环境变量{Colors.ENDC}")
            return

        print(f"\n{Colors.BOLD}=== 清除 Redis 缓存 ==={Colors.ENDC}")
        print(f"Redis URL: {RAG_REDIS_URL}")
        print(f"Cache Name: {LLAMA_REDIS_CACHE_NAME}")
        print(f"DocStore Name: {LLAMA_DOC_STORE_NAME}")

        try:
            r = redis.from_url(RAG_REDIS_URL, decode_responses=True)

            # 统计当前键数量
            cache_keys = list(r.scan_iter(f"{LLAMA_REDIS_CACHE_NAME}*"))
            docstore_keys = list(r.scan_iter(f"{LLAMA_DOC_STORE_NAME}*"))

            print(f"\n当前状态:")
            print(f"  - Cache 键数量: {len(cache_keys)}")
            print(f"  - DocStore 键数量: {len(docstore_keys)}")

            if len(cache_keys) == 0 and len(docstore_keys) == 0:
                print(f"{Colors.WARNING}缓存为空，无需清理{Colors.ENDC}")
                return

            confirm = input(f"\n{Colors.WARNING}确认清除所有缓存? (y/n): {Colors.ENDC}")
            if confirm.lower() != "y":
                print("操作已取消")
                return

            # 删除所有缓存键
            deleted_cache = 0
            for key in cache_keys:
                r.delete(key)
                deleted_cache += 1

            # 删除所有 docstore 键
            deleted_docstore = 0
            for key in docstore_keys:
                r.delete(key)
                deleted_docstore += 1

            print(f"{Colors.GREEN}✓ 清除完成!{Colors.ENDC}")
            print(f"  - 已删除 Cache 键: {deleted_cache}")
            print(f"  - 已删除 DocStore 键: {deleted_docstore}")

        except Exception as e:
            print(f"{Colors.FAIL}清除失败: {e}{Colors.ENDC}")

    def clear_collection_data(self):
        """清除指定集合的所有数据（保留集合结构）"""
        cols = self.list_collections()
        if not cols:
            return

        col = self.select_collection()
        if not col:
            return

        count = col.count()
        if count == 0:
            print(f"{Colors.WARNING}集合 {col.name} 为空，无需清理{Colors.ENDC}")
            return

        print(f"\n{Colors.BOLD}=== 清除集合数据 ==={Colors.ENDC}")
        print(f"集合名称: {col.name}")
        print(f"当前文档数量: {count}")

        confirm = input(f"\n{Colors.WARNING}确认清除所有数据? (y/n): {Colors.ENDC}")
        if confirm.lower() != "y":
            print("操作已取消")
            return

        try:
            # 获取所有 ID
            results = col.get(include=["ids"])
            all_ids = results["ids"]

            # 批量删除（Chroma 建议每次最多删除 5000 个）
            batch_size = 5000
            for i in range(0, len(all_ids), batch_size):
                batch = all_ids[i:i + batch_size]
                col.delete(ids=batch)

            print(f"{Colors.GREEN}✓ 已删除 {len(all_ids)} 个文档{Colors.ENDC}")

        except Exception as e:
            print(f"{Colors.FAIL}删除失败: {e}{Colors.ENDC}")

    def clear_all_caches(self):
        """清除所有缓存（Redis + ChromaDB 集合数据）"""
        print(f"\n{Colors.FAIL}{Colors.BOLD}!!! 清除所有缓存 !!!{Colors.ENDC}")
        print("这将清除:")
        print("  1. Redis Ingestion Cache")
        print("  2. Redis DocStore")
        print("  3. 所有 ChromaDB 集合中的数据")

        confirm = input("\n请输入 'CLEAR' 确认操作: ")
        if confirm != "CLEAR":
            print("操作已取消")
            return

        print(f"\n{Colors.BOLD}步骤 1/3: 清除 Redis 缓存{Colors.ENDC}")
        self.clear_redis_cache()

        print(f"\n{Colors.BOLD}步骤 2/3: 清除 ChromaDB 数据{Colors.ENDC}")
        cols = self.client.list_collections()
        total_deleted = 0
        for col in cols:
            real_col = self.client.get_collection(col.name)
            count = real_col.count()
            if count > 0:
                print(f"  - 清除 {col.name} ({count} 个文档)...")
                try:
                    results = real_col.get(include=["ids"])
                    all_ids = results["ids"]
                    batch_size = 5000
                    for i in range(0, len(all_ids), batch_size):
                        batch = all_ids[i:i + batch_size]
                        real_col.delete(ids=batch)
                    total_deleted += len(all_ids)
                except Exception as e:
                    print(f"    {Colors.FAIL}失败: {e}{Colors.ENDC}")

        print(f"\n{Colors.BOLD}步骤 3/3: 完成{Colors.ENDC}")
        print(f"{Colors.GREEN}✓ 总计删除 {total_deleted} 个文档{Colors.ENDC}")


def main():
    admin = ChromaAdmin()

    while True:
        print(f"\n{Colors.HEADER}=== ChromaDB 运维工具箱 ==={Colors.ENDC}")
        print("1. 查看集合列表 (List)")
        print("2. 预览数据 (Peek Data) -- 检查写入情况")
        print("3. RAG 检索测试 (Query) -- 检查召回效果")
        print("4. 删除集合 (Delete Collection)")
        print("5. 清空整个数据库 (Reset All)")
        print("6. 清除 Redis 缓存 (Clear Redis Cache) -- 清除 Ingestion Cache 和 DocStore")
        print("7. 清除集合数据 (Clear Collection Data) -- 保留集合结构，仅清除数据")
        print("8. 清除所有缓存 (Clear All) -- Redis + ChromaDB 数据")
        print("0. 退出")

        choice = input("\n请选择功能: ")

        if choice == "1":
            admin.list_collections()
        elif choice == "2":
            admin.peek_data()
        elif choice == "3":
            admin.rag_test()
        elif choice == "4":
            admin.delete_collection()
        elif choice == "5":
            admin.reset_db()
        elif choice == "6":
            admin.clear_redis_cache()
        elif choice == "7":
            admin.clear_collection_data()
        elif choice == "8":
            admin.clear_all_caches()
        elif choice == "0":
            sys.exit(0)
        else:
            print("无效输入")


if __name__ == "__main__":
    main()
