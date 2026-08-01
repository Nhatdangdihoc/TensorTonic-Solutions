def validate_records(records, schema):
    """
    Validate records against a schema definition.

    records: list[dict] — các bản ghi cần kiểm tra
    schema: list[dict] — mỗi phần tử có các key:
        - "column": tên cột (str)
        - "type": "int" | "float" | "str"
        - "nullable": bool (mặc định coi như True nếu không có)
        - "min", "max": giới hạn số (tùy chọn)

    Returns: list[(record_index, is_valid, errors)]
        - record_index: chỉ số 0-based của record
        - is_valid: True nếu không có lỗi
        - errors: danh sách chuỗi lỗi (rỗng nếu hợp lệ)
    """
    type_map = {"int": int, "float": float, "str": str}
    results = []

    for idx, record in enumerate(records):
        errors = []

        for col_def in schema:
            col = col_def["column"]
            expected_type_name = col_def["type"]
            nullable = col_def.get("nullable", True)
            min_val = col_def.get("min")
            max_val = col_def.get("max")

            # 1. Missing column check
            if col not in record:
                errors.append(f"{col}: missing")
                continue

            value = record[col]

            # 2. Null check
            if value is None:
                if not nullable:
                    errors.append(f"{col}: null")
                continue  # None -> bỏ qua type/range check dù nullable hay không

            # 3. Type check
            actual_type = type(value)
            if expected_type_name == "float":
                # float chấp nhận cả int, nhưng KHÔNG chấp nhận bool
                type_ok = actual_type in (int, float)
            else:
                type_ok = actual_type is type_map[expected_type_name]

            if not type_ok:
                errors.append(f"{col}: expected {expected_type_name}, got {actual_type.__name__}")
                continue  # type sai -> bỏ qua range check

            # 4. Range check (chỉ khi min/max được định nghĩa và giá trị là số)
            if (min_val is not None or max_val is not None) and actual_type in (int, float):
                if (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
                    errors.append(f"{col}: out of range")

        results.append((idx, len(errors) == 0, errors))

    return results