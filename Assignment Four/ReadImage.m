function image = ReadImage(group, X, k)
Img = [];
pos = find(group == k);
ind = randperm(length(pos), 20);
for l = ind
    Img = [Img reshape(X(pos(l), :), 28, 28)];
end

image = imshow(Img);

end