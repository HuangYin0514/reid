from data.build import make_data_loader


class init_opt:
    img_height = 4
    img_width = 2
    batch_size = 10
    test_batch_size = 20


opt = init_opt()

if __name__ == "__main__":
    train_loader, val_loader, len_query, num_classes = make_data_loader(
        opt, "market1501", "/Users/huangyin/Documents/datasets/Market-1501-v15.09.15"
    )

    for i in train_loader:
        data, label = i
        print(data.shape, label)
        break

    for i in val_loader:
        data, label = i
        print(data.shape, label)
        break
