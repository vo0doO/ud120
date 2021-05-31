#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Удалите 10% точек с наибольшим
        остаточные ошибки (разница между прогнозом
        и фактическая чистая стоимость).

        Вернуть список кортежей с именем cleaned_data, где
        каждый кортеж имеет форму (возраст, чистая_доходность, ошибка).
    """
    
    cleaned_data = []

    ### ваш код идет сюда
    max_errors = -80
    count = 0
    max_count = len(predictions)
    while count < max_count:

        errors = net_worths[count][0]-predictions[count][0]

        if errors < max_errors:
            count = count + 1
            continue

        cleaned_data.append((ages[count][0], net_worths[count][0], errors))
        count = count + 1

    return cleaned_data

