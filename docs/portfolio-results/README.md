# JEPA-GLM: 실험 결과와 진행 상태

이 문서는 공개본에 보존된 실험·검증 기록을 시각화한 것이다. 원래 학습·과학 실험을 새로 수행했다는 의미는 아니다. 기능 테스트, 합성 데모, 실제 성능 지표를 서로 구분한다. 진행률의 임의 퍼센트는 사용하지 않는다.

![실험 및 검증 요약](results.png)

## 결과 해석

Local task의 작은 양의 차이는 외부 과제에서 일관되게 유지되지 않았다. 왼쪽 구간은 단일 seed의 사후 구간이며, 오른쪽은 고정된 10,000-variant 평가의 seed SD다. 두 종류의 구간을 같은 통계량으로 읽으면 안 된다.

## 현재 진행 상태

| 항목 | 확인된 상태 |
|---|---|
| JEPA 학습·평가 코드 | 구현 및 기존 비교 결과 보유 |
| 공개본 검증 | 26개 테스트, CPU synthetic 8-step 학습 완료 |
| 대규모 재학습 | 이번 문서 작업에서 미실행 |
| 일관된 downstream 개선 | 현재 기록만으로 확정하지 않음 |

## 다음 보완 과제

1. 선택 과정과 분리된 seed·task 재현 설계
2. 동일한 학습 예산의 MLM 비교 및 불확실성 보고
3. 표현 손실 감소와 생물학적 예측 성능 개선의 관계 검토

## 보안 범위와 남은 검증

모델 키·학습 checkpoint·원시 유전체는 공개하지 않았다. 실제 입력의 민감성과 외부 모델 download·학습 결과 공개 범위는 별도로 관리해야 한다.
`.gitignore` 외에 공개 파일 내용도 검사했다. 이전에 유출된 비밀정보를 ignore 규칙만으로 회수할 수는 없다.

## 근거와 그림 재현

- [outputs/comprehensive_negative_figures/figure1_plot_data.csv](../../outputs/comprehensive_negative_figures/figure1_plot_data.csv)
- [VALIDATION.md](../../VALIDATION.md)
- [그림의 수치와 조건](metrics.json)
- [확대 가능한 SVG](results.svg)
- [그림 재생성 코드](reproduce_figures.py)

```bash
python -m pip install matplotlib
python docs/portfolio-results/reproduce_figures.py
```

원시 실험 재현은 각 프로젝트의 본문 프로토콜을 따른다. 위 명령은 보존된 수치로 그림만 다시 만든다.
