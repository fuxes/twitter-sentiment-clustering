(function() {
  'use strict';

  angular
    .module('bubbleApp.tweetlist')
    .controller('TweetListCtrl', TweetListCtrl);

  TweetListCtrl.$inject = [
    '$scope',
    'Bubbles',
    'Tweets'
  ];

  function TweetListCtrl($scope, Bubbles, Tweets) {
    var vm = this;
    vm.tweetsIds = [];
    vm.activeTerm = false;
    vm.activeCluster =0;

    Bubbles.onBubbleClick(onBubbleClick);

    activate();

    function activate() {
      getTweets();
    }

    function getTweets() {
      return Tweets.get().then(function(tweets) {
        vm.tweets = tweets;
      });
    }

    function onBubbleClick(tweet) {
      vm.activeTerm = tweet.term;
      vm.tweetsIds = _.uniq(tweet.tweets);

      $scope.$apply();
    }
  }
})();