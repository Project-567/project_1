# Results

## Policy Evaluation with Even Policy:

Discount Factor = 0.99

Iterations: 929

Value Function: 
[[ 3.2175  7.1243  4.3031  4.7526  1.5245]
 [ 1.4609  2.7247  2.2163  1.7565  0.3782]
 [-0.4904  0.2073  0.1705 -0.2499 -1.1211]
 [-2.1493 -1.5671 -1.4849 -1.8157 -2.5267]
 [-3.467  -2.9047 -2.7873 -3.0748 -3.7354]]

Discount Factor = 0.95

Iterations: 183

Value Function: 
[[ 3.6876  8.3679  4.7801  5.4066  1.8618]
 [ 1.8888  3.2641  2.6248  2.2025  0.8142]
 [ 0.165   0.8619  0.8052  0.428  -0.3975]
 [-1.1684 -0.6053 -0.5243 -0.8081 -1.4659]
 [-2.2581 -1.718  -1.5995 -1.8403 -2.448 ]]

 Discount Factor = 0.8

 Iterations: 44

 Value Function: 
[[ 2.5435  9.2384  3.6861  5.1635  0.9447]
 [ 0.8922  2.4403  1.5926  1.4262  0.1705]
 [-0.165   0.4781  0.4105  0.2043 -0.4387]
 [-0.7803 -0.2954 -0.2226 -0.3762 -0.8798]
 [-1.4108 -0.952  -0.8519 -0.9831 -1.4543]]


## Policy Evaluation with Random Policy:

- Running policy evaluation for random policy 5 times (each time gives a different value funciton)

Discount Factor = 0.99

Iterations: 1206

Value Function: 
[[-14.0167  -9.5499 -11.9047 -11.4114 -17.5361]
 [-18.0197 -15.1561 -14.5094 -14.6683 -17.0314]
 [-18.1638 -15.6521 -15.586  -16.5772 -18.0159]
 [-18.5879 -17.2969 -17.0633 -17.7182 -18.6638]
 [-19.9172 -19.7474 -18.4444 -18.6228 -19.3389]]

Iterations: 1072

Value Function: 
[[11.8312 13.633   8.8724 10.3177  5.7956]
 [ 9.3358 10.5437  8.3861  7.4159  4.3389]
 [ 7.9897  8.8286  6.7144  5.3714  3.5319]
 [ 5.2591  5.8413  5.4346  3.4257  1.9086]
 [ 3.3675  3.6697  2.9873  2.2453  1.2961]]

Iterations: 1167

Value Function: 
[[ -8.3948  -4.5135  -6.1531  -3.3165  -4.9283]
 [-12.8125  -9.2545  -7.7724  -6.1907  -5.7237]
 [-13.7692 -12.2347 -10.0599  -8.4005  -7.5457]
 [-14.5235 -13.9587 -10.7528  -9.8337  -8.9965]
 [-15.1191 -14.6601 -12.6359 -12.4388 -11.9126]]

Iterations: 1188

Value Function: 
[[-13.1398  -7.743   -8.7569  -4.9099  -9.2627]
 [-14.1328 -11.237   -9.766   -8.4009  -9.2979]
 [-15.5861 -14.1474 -10.9229 -10.01   -10.2303]
 [-18.4548 -17.1182 -13.9947 -12.6424 -11.5765]
 [-21.6855 -17.9222 -15.2629 -13.7196 -14.075 ]]

Iterations: 1077

Value Function: 
[[-3.1144  4.2356  2.0821  1.8595 -0.5073]
 [-4.3726 -1.6629 -0.1557 -0.3776 -2.2246]
 [-5.8668 -5.5925 -3.2689 -3.1723 -3.9021]
 [-6.9732 -6.0432 -3.9745 -4.0475 -4.7527]
 [-6.7039 -5.8226 -5.0291 -5.3256 -6.4919]]

## Policy Iteration with Random Policy, Discount Factor 0.99:

Iterations: 2989

Policy Stopped Changing: 2988

Value Map: 
[[201.9998 204.0402 201.9998 199.0402 197.0498]
 [199.9798 201.9998 199.9798 197.98   196.0002]
 [197.98   199.9798 197.98   196.0002 194.0402]
 [196.0002 197.98   196.0002 194.0402 192.0998]
 [194.0402 196.0002 194.0402 192.0998 190.1788]]

Policy Table: 
       0   1     2     3     4
0  right  up  left    up  left
1     up  up    up  left  left
2     up  up    up    up    up
3     up  up    up    up    up
4     up  up    up    up    up

## Policy Iteration with Random Policy, Discount Factor 0.95:

Iterations: 874

Policy Stopped Changing: 873

Value Map: 
[[41.9947 44.2049 41.9947 39.2049 37.2447]
 [39.895  41.9947 39.895  37.9002 36.0052]
 [37.9002 39.895  37.9002 36.0052 34.2049]
 [36.0052 37.9002 36.0052 34.2049 32.4947]
 [34.2049 36.0052 34.2049 32.4947 30.87  ]]

Policy Table: 
       0   1     2     3     4
0  right  up  left    up  left
1     up  up    up  left  left
2     up  up    up    up    up
3     up  up    up    up    up
4     up  up    up    up    up

## Policy Iteration with Random Policy, Discount Factor 0.8:

Iterations: 140

Policy Stopped Changing: 139

Value Map: 
[[11.8991 14.8739 11.8991 10.2459  8.1967]
 [ 9.5193 11.8991  9.5193  8.1967  6.5574]
 [ 7.6154  9.5193  7.6154  6.5574  5.2459]
 [ 6.0923  7.6154  6.0923  5.2459  4.1967]
 [ 4.8739  6.0923  4.8739  4.1967  3.3574]]

Policy Table: 
       0   1     2   3     4
0  right  up  left  up  left
1     up  up    up  up    up
2     up  up    up  up    up
3     up  up    up  up    up
4     up  up    up  up    up


## Value Iteration with Random Policy, Discount Factor 0.99:

Iterations: 1605

Value Map: 
[[201.9998 204.0402 201.9998 199.0402 197.0498]
 [199.9798 201.9998 199.9798 197.98   196.0002]
 [197.98   199.9798 197.98   196.0002 194.0402]
 [196.0002 197.98   196.0002 194.0402 192.0998]
 [194.0402 196.0002 194.0402 192.0998 190.1788]]

Policy Table: 
       0   1     2     3     4
0  right  up  left    up  left
1     up  up    up  left  left
2     up  up    up    up    up
3     up  up    up    up    up
4     up  up    up    up    up



## Value Iteration with Random Policy, Discount Factor 0.95:

Iterations: 316

Value Map: 
[[41.9947 44.2049 41.9947 39.2049 37.2447]
 [39.895  41.9947 39.895  37.9002 36.0052]
 [37.9002 39.895  37.9002 36.0052 34.2049]
 [36.0052 37.9002 36.0052 34.2049 32.4947]
 [34.2049 36.0052 34.2049 32.4947 30.87  ]]

Policy Table: 
       0   1     2     3     4
0  right  up  left    up  left
1     up  up    up  left  left
2     up  up    up    up    up
3     up  up    up    up    up
4     up  up    up    up    up


## Value Iteration with Random Policy, Discount Factor 0.8:

Iterations: 74

Value Map: 
[[11.8991 14.8739 11.8991 10.2459  8.1967]
 [ 9.5193 11.8991  9.5193  8.1967  6.5574]
 [ 7.6154  9.5193  7.6154  6.5574  5.2459]
 [ 6.0923  7.6154  6.0923  5.2459  4.1967]
 [ 4.8739  6.0923  4.8739  4.1967  3.3574]]

Policy Table: 
       0   1     2   3     4
0  right  up  left  up  left
1     up  up    up  up    up
2     up  up    up  up    up
3     up  up    up  up    up
4     up  up    up  up    up












