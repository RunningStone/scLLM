
def mask_list(original_list,mask):
    """
    Mask a list with a mask
    :param original_list: list to be masked
    :param mask: mask with True/False values [True, False, True, ...]
    :return: masked list
    """
    return [x for x, m in zip(original_list, mask) if m]