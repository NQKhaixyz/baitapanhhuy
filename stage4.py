"""Chặng 4 — chạy eval định lượng và ghi artifact có timestamp."""

from __future__ import annotations

from eval import main


if __name__ == "__main__":
    # Stage4 là entrypoint mỏng để người học có thể gọi đúng tên chặng;
    # logic chấm nằm tập trung ở eval.py để tái sử dụng khi CI chạy.
    main()
