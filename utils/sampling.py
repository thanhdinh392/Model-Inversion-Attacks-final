import numpy as np


# Using for mnist and fashion-mnist
def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = {}
    num_items = int(len(dataset) / num_users)
    all_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# mnist non-iid even to odd (version 2)
# def mnist_noniid(dataset, num_users):
#     """
#     Chia dữ liệu MNIST thành non-I.I.D shards
#     Mỗi class sẽ được chia làm 1 shard và đảm bảo mỗi client có 2 class khác nhau
#     :param dataset: Dataset MNIST
#     :param num_users: Số lượng người dùng (clients)
#     :return: dict_users: Dictionary chứa các shard phân chia cho từng client
#     """
#     # Khởi tạo
#     dict_users = {}
#     num_shards = num_users * 2  # 5 clients * 2 = 10 shards
#     num_classes = 10  # MNIST có 10 class
#     labels = dataset.train_labels.numpy()  # Lấy nhãn của dataset
#     idxs = np.arange(len(dataset))  # Chỉ mục của toàn bộ dữ liệu
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

#     # Sắp xếp các ảnh theo nhãn
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]  # Lấy các chỉ mục đã được sắp xếp theo nhãn

#     # Tạo danh sách các shards
#     list_shards = [[] for _ in range(num_shards)]

#     # Số ảnh mỗi shard sẽ nhận (chia đều mỗi class làm 1 shard)
#     for class_id in range(num_classes):
#         class_idxs = idxs_labels[0, idxs_labels[1] == class_id]  # Lấy các ID thuộc class đó

#         # Gán toàn bộ ảnh của class này cho shard tương ứng
#         list_shards[class_id].extend(class_idxs)

#     # Phân chia shards :
#     # client1: shards 0, 1
#     # client2: shards 2, 3
#     # client3: shards 4, 5
#     # client4: shards 6, 7
#     # client5: shards 8, 9
#     for i in range(num_users):
#         shards_for_user = [i * 2, i * 2 + 1]  # Mỗi client nhận 2 shards (tương ứng với 2 class)
#         for shard in shards_for_user:
#             dict_users[i] = np.concatenate((dict_users[i], list_shards[shard]), axis=0)
#     return dict_users


# Using for mnist and fashion-mnist
# mnist non-iid even to odd (version 4)
def mnist_noniid(dataset, num_users):
    """
    Chia dữ liệu MNIST thành non-I.I.D shards
    Mỗi class sẽ được chia làm 2 shards và đảm bảo mỗi client có 4 class khác nhau
    :param dataset: Dataset MNIST
    :param num_users: Số lượng người dùng (clients)
    :return: dict_users: Dictionary chứa các shard phân chia cho từng client
    """
    # Khởi tạo
    dict_users = {}
    num_shards = num_users * 4  # 5 clients * 4 = 20 shards
    num_classes = 10  # MNIST có 10 class
    labels = dataset.train_labels.numpy()  # Lấy nhãn của dataset
    idxs = np.arange(len(dataset))  # Chỉ mục của toàn bộ dữ liệu
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # Sắp xếp các ảnh theo nhãn
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]  # Lấy các chỉ mục đã được sắp xếp theo nhãn

    # Tạo danh sách các shards
    list_shards = [[] for _ in range(num_shards)]

    # Số ảnh mỗi shard sẽ nhận (chia đều mỗi class làm 2 phần)
    for class_id in range(num_classes):
        class_idxs = idxs_labels[0, idxs_labels[1] == class_id]  # Lấy các ID thuộc class đó
        half = len(class_idxs) // 2  # Chia đôi class

        # Phân chia ID của class thành 2 shards
        shard_1 = class_idxs[:half]
        shard_2 = class_idxs[half:]

        # Gán các shard này cho các vị trí shard tương ứng
        list_shards[class_id * 2].extend(shard_1)
        list_shards[class_id * 2 + 1].extend(shard_2)

    # Phân chia shards theo yêu cầu
    even_shards = [i for i in range(0, num_shards, 2)]  # Chỉ số chẵn
    odd_shards = [i for i in range(1, num_shards, 2)]  # Chỉ số lẻ

    # Phân các shard chẵn trước cho các client
    shard_assignments = even_shards[:num_users * 2] + odd_shards[:num_users * 2]

    for i in range(num_users):
        # Gán shards chẵn cho client đầu tiên
        shards_for_user = shard_assignments[i * 4:(i + 1) * 4]
        for shard in shards_for_user:
            dict_users[i] = np.concatenate((dict_users[i], list_shards[shard]), axis=0)

    return dict_users

# Divide version 1 client 4 classes random (long time)
# def mnist_noniid(dataset, num_users):
#     """
#     Chia dữ liệu MNIST thành non-I.I.D shards
#     Mỗi class sẽ được chia làm 2 shards và đảm bảo mỗi client có 4 class khác nhau
#     :param dataset: Dataset MNIST
#     :param num_users: Số lượng người dùng (clients)
#     :return: dict_users: Dictionary chứa các shard phân chia cho từng client
#     """
#     # Khởi tạo
#     dict_users = {}
#     num_shards = num_users * 4  # 5 clients * 4 = 20 shards
#     num_classes = 10  # MNIST có 10 class
#     labels = dataset.train_labels.numpy()  # Lấy nhãn của dataset
#     idxs = np.arange(len(dataset))  # Chỉ mục của toàn bộ dữ liệu
#     idx_shard = [i for i in range(num_shards)]

#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

#     # Sắp xếp các ảnh theo nhãn
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]  # Lấy các chỉ mục đã được sắp xếp theo nhãn

#     # Tạo danh sách các shards
#     list_shards = [[] for _ in range(num_shards)]

#     # Số ảnh mỗi shard sẽ nhận (chia đều mỗi class làm 2 phần)
#     for class_id in range(num_classes):
#         class_idxs = idxs_labels[0, idxs_labels[1] == class_id]  # Lấy các ID thuộc class đó
#         half = len(class_idxs) // 2  # Chia đôi class

#         # Phân chia ID của class thành 2 shards
#         shard_1 = class_idxs[:half]
#         shard_2 = class_idxs[half:]

#         # Gán các shard này cho các vị trí shard tương ứng
#         if class_id * 2 < num_shards:
#             list_shards[class_id * 2].extend(shard_1)
#         if class_id * 2 + 1 < num_shards:
#             list_shards[class_id * 2 + 1].extend(shard_2)

#     # Phân chia shards cho các client sao cho mỗi client có 4 class khác nhau
#     for i in range(num_users):
#         chosen_classes = set()
#         user_shards = []

#         # Chọn 4 shards sao cho các shard thuộc 4 class khác nhau
#         while len(chosen_classes) < 4:
#             rand_shard = np.random.choice(idx_shard, 1, replace=False)[0]
#             class_id = rand_shard // 2  # Mỗi class có 2 shards, shard 0 và 1 thuộc class 0, shard 2 và 3 thuộc class 1, ...

#             if class_id not in chosen_classes:
#                 chosen_classes.add(class_id)
#                 user_shards.append(rand_shard)
#                 idx_shard.remove(rand_shard)  # Loại bỏ shard đã chọn khỏi danh sách

#         # Gán dữ liệu từ các shards đã chọn cho user i
#         for shard in user_shards:
#             dict_users[i] = np.concatenate((dict_users[i], list_shards[shard]), axis=0)
#     return dict_users


# Divide origin (bad)
# def mnist_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     dict_users = {}
#     num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#     idxs = np.arange(num_shards * num_imgs)
#     labels = dataset.train_labels.numpy()
#
#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]
#
#     # divide and assign
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
#     return dict_users

