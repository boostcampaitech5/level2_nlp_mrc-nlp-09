import os
import json


def hard_voting():
    path = "../ensemble/hard"
    ensemble_lists = os.listdir(path)

    # 예측값들을 리스트에 저장
    answers = []
    for ensemble_list in ensemble_lists:
        with open(os.path.join(path, ensemble_list), "r") as f:
            answers.append(json.load(f))

    # 각 데이터별 예측값들을 리스트에 저장
    keys = list(answers[0].keys())
    result = []
    for i in range(len(answers[0])):
        temp = []
        for j in range(len(answers)):
            temp.append(answers[j][keys[i]])
        result.append(temp)

    # 빈도수 기반 투표
    final_result = {}
    count = 0
    for i in range(len(result)):
        temp = {}
        for j in range(len(result[i])):
            if result[i][j] in temp:
                temp[result[i][j]] += 1
            else:
                temp[result[i][j]] = 1
        final_result[keys[i]] = max(temp, key=temp.get)
        max_count = max(temp.values())
        for key, value in temp.items():
            if value == max_count and key != final_result[keys[i]]:
                print("동률 발생:", keys[i])
                print(temp)
                print(f"선택된 정담: {final_result[keys[i]]}")
                print(f"동률 단어: {key}")
                print()
                count += 1
    print(f"동률 개수: {count}")

    # 결과 저장
    with open("../ensemble/ensemble_hard.json", "w") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)


def soft_voting():
    path = "../ensemble/soft"
    ensemble_lists = os.listdir(path)

    # 예측값들을 리스트에 저장
    answers = []
    for ensemble_list in ensemble_lists:
        with open(os.path.join(path, ensemble_list), "r") as f:
            answers.append(json.load(f))

    # 각 데이터별 예측값들을 확률과 함께 리스트에 저장
    keys = list(answers[0].keys())
    result = []
    for i in range(len(answers[0])):  # 600
        temp = []
        for j in range(len(answers)):
            for k in range(5):
                # 예측값이 이미 있으면 확률을 더해주고, 없으면 새로 추가
                found = False
                for index, value in enumerate(temp):
                    if value[0] == answers[j][keys[i]][k]["text"]:
                        temp[index][1] += answers[j][keys[i]][k]["probability"]
                        found = True
                        break
                if not found:
                    temp.append(
                        [answers[j][keys[i]][k]["text"],
                         answers[j][keys[i]][k]["probability"]]
                    )
        result.append(temp)
        temp.sort(key=lambda x: x[1], reverse=True)

    # 확률 기반 투표
    final_result = {}
    for i in range(len(result)):
        final_result[keys[i]] = result[i][0][0]

    # 결과 저장
    with open("../ensemble/ensemble_soft.json", "w") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    hard_voting()
    # soft_voting()
